from pathlib import Path
import tqdm.auto as tqdm
import xarray as xr, pandas as pd
import dask.array as da
import anyio, shutil
import heapq, json
from contextlib import asynccontextmanager
from typing import Callable, Awaitable, Any, TypeVar, List, Dict, Union, Literal
import logging
import anyio.to_process
import anyio.to_thread
import functools
import pickle

logger = logging.getLogger(__name__)

class Limiter:
    """A priority-based async concurrency limiter built with anyio primitives.

    Example:
        limiter = Limiter(max_concurrent=5)
        async with limiter.get(priority=10):
            await do_work()
    """

    def __init__(self, max_concurrent: int):
        self._max_concurrent = max_concurrent
        self._current = 0
        self._waiters = []  # heap of (-priority, counter, event)
        self._counter = 0   # tie-breaker for FIFO ordering
        self._lock = anyio.Lock()

    @asynccontextmanager
    async def get(self, priority: float = 0.0):
        """Acquire a slot respecting priority (lower = sooner)."""
        await self._acquire(priority)
        try:
            yield
        finally:
            await self._release()

    async def _acquire(self, priority: float):
        async with self._lock:
            if self._current < self._max_concurrent and not self._waiters:
                self._current += 1
                return

            # Need to wait — register our waiter
            event = anyio.Event()
            heapq.heappush(self._waiters, (priority, self._counter, event))
            self._counter += 1

        # Wait for release to signal our event
        await event.wait()
        # Once awakened, we've been granted a slot
        return

    async def _release(self):
        async with self._lock:
            self._current -= 1
            if self._waiters:
                # Give the slot to the next highest-priority waiter
                _, _, event = heapq.heappop(self._waiters)
                self._current += 1
                event.set()

groups = {}
my_limiter = Limiter(5)
base_result_path: Path = None
# Path("/media/julienb/T7 Shield/Revue-FRM/AnalysisData2/")

def set_limiter(limit: 5):
    global my_limiter
    my_limiter = Limiter(limit)

def set_base_result_path(p: Path):
    global base_result_path
    base_result_path = Path(p)

def add_note_to_exception(exc, note):
    #Because we are currently using python 3.9 and we dont have add_note
    new_exc = exc.__class__(f"{exc}\nNote: {note}")
    new_exc.__cause__ = exc.__cause__
    new_exc.__context__ = exc.__context__
    return new_exc


def in_runner_func(func, tmp_path, save_fn):
    res = func()
    tmp_path.parent.mkdir(exist_ok=True, parents=True)
    if tmp_path.exists():
        if tmp_path.is_dir():
            shutil.rmtree(tmp_path)
        else:
            tmp_path.unlink()
    save_fn(res, tmp_path)

from multiprocessing.reduction import ForkingPickler
import io

import multiprocessing as mp

def _roundtrip_in_subprocess(data):
    import pickle
    pickle.loads(data)  # just try to unpickle
    return True

def test_anyio_picklable(obj):
    buf = io.BytesIO()
    ForkingPickler(buf).dump(obj)
    data = buf.getvalue()

    ctx = mp.get_context("spawn")  # AnyIO uses "spawn" by default
    with ctx.Pool(1) as pool:
        try:
            pool.apply(_roundtrip_in_subprocess, (data,))
            return True
        except Exception as e:
            print(f"❌ Not picklable for AnyIO: {type(e).__name__}: {e}")
            raise

T = TypeVar("T")
def mk_checkpoint(
    save_fn: Callable[[T, Path], None],
    load_fn: Callable[[Path], T]
) -> Callable[
    [Callable[[], T], Path, str, float],
    Awaitable[Callable[[], T]]
]:
    async def checkpoint(func, path: Path, group: str, priority: float, mode: Literal["thread", "process"] = "thread"):
        id = str(path)
        path = base_result_path/path
        #No need for thread safety, this should always be called from the main thread
        if not group in groups:
            groups[group] = [tqdm.tqdm(desc=group, total=0), 0, 0]
        bar = groups[group][0]
        if not path.exists():
            bar.total+=1
            bar.refresh()
            tmp_path = path.with_name(".tmp"+path.name)
            #Its fine if the tmp file lingers, it can be used for bug inspection
            async with my_limiter.get(priority):
                groups[group][2]+=1
                bar.set_postfix(dict(prev_done=groups[group][1], computing=groups[group][2]))
                mrun = functools.partial(in_runner_func, func, tmp_path, save_fn)
                try:
                    if mode == "thread":
                        await anyio.to_thread.run_sync(mrun)
                    elif mode=="process":
                        await anyio.to_process.run_sync(mrun)
                except Exception as e:
                    # logger.exception(f"While computing {id}")
                    raise add_note_to_exception(e, f"While computing {id}")
                finally:
                    groups[group][2]-=1
                    bar.set_postfix(dict(prev_done=groups[group][1], computing=groups[group][2]))
            shutil.move(tmp_path, path)
            bar.update(1)
            bar.refresh()
        else:
            groups[group][1]+=1
            bar.set_postfix(dict(prev_done=groups[group][1], computing=groups[group][2]))
        return lambda : load_fn(path)
    return checkpoint

def _load_xarray(path: Path) -> Union[xr.DataArray, xr.Dataset]: 
    obj = xr.open_zarr(path)
    # if  obj:
    #     raise Exception("Load result is None...")
    for k in list(obj.data_vars) + list(obj.coords):
        o: xr.Dataset = obj[k]
        if obj[k].attrs.pop("__computed_"):
            obj[k] = obj[k].compute()
        o.encoding.pop("filters", None)

    if obj.attrs.pop("_is_xrdatarray_"):
        vars = list(obj.data_vars)
        if len(vars) != 1:
            raise Exception("Problem")
        obj = obj[vars[0]]
    return obj

def _save_xarray(obj: Union[xr.DataArray, xr.Dataset], path: Path):
    if isinstance(obj, xr.DataArray):
        if obj.name is None:
            obj.name = "data"
        obj = obj.to_dataset()
        obj.attrs["_is_xrdatarray_"] = True
    else:
        obj.attrs["_is_xrdatarray_"] = False
    obj:xr.Dataset
    encoding = {}
    for k in list(obj.data_vars) + list(obj.coords):
        ar: xr.DataArray = obj[k]
        is_computed = not isinstance(ar.data, da.Array)
        obj[k].attrs["__computed_"] = is_computed
        obj[k].encoding.pop("filters", None)
        encoding[k] = {"compressor": None, "chunks": ar.shape if is_computed else ar.data.chunksize}
    # import time
    # start = time.time()
    obj.to_zarr(path, compute=True, encoding=encoding, mode="w")#compute=True doesnt do anything on non chunked arrays
    # print(f"duration={time.time()-start}s")

checkpoint_xarray = mk_checkpoint(_save_xarray, _load_xarray)

def _save_json(obj, path: Path):
    with path.open("w") as f:
        json.dump(obj, f)

def _load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)

checkpoint_json =  mk_checkpoint(_save_json, _load_json)

def _save_excel(obj: pd.DataFrame, path: Path):
    obj.to_excel(path)

def _load_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)

checkpoint_excel =  mk_checkpoint(_save_excel, _load_excel)

lock = anyio.Lock()
do_stop = False

class _Single:
    def __init__(self, lock: anyio.Lock, stop: anyio.Event):
        self.lock, self.stop = lock, stop

    async def __aenter__(self):
        await self.lock.acquire()
        if self.stop.is_set():
            self.lock.release()
            # Wait indefinitely, but cooperatively cancellable
            await anyio.Event().wait()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if exc_type:
                self.stop.set()
        finally:
            self.lock.release()

_lock = anyio.Lock()
_event_lock = anyio.Event()

@asynccontextmanager
async def _noop_async_cm():
    yield

def single(yes: bool = True):
    if yes:
        return _Single(_lock, _event_lock)
    else:
        return _noop_async_cm()

# async def checkpoint_xarray(func, path: Path, group: str, priority: float):
#     id = str(path)
#     path = base_result_path/path
#     #No need for thread safety, this should always be called from the main thread
#     if not group in groups:
#         groups[group] = [tqdm.tqdm(desc=group, total=0), 0, 0]
#     bar = groups[group][0]
#     if not path.exists():
#         bar.total+=1
#         bar.refresh()
#         tmp_path = path.with_name(".tmp"+path.name)
#         #Its fine if the tmp file lingers, it can be used for bug inspection
#         async with my_limiter.get(priority):
#             groups[group][2]+=1
#             bar.set_postfix(dict(prev_done=groups[group][1], computing=groups[group][2]))
#             def mrun():
#                 res = func()
#                 _save(res, tmp_path)
#             try:
#                 await anyio.to_thread.run_sync(mrun)
#             except Exception as e:
#                 raise add_note_to_exception(e, f"While computing {id}")
#             finally:
#                 groups[group][2]-=1
#                 bar.set_postfix(dict(prev_done=groups[group][1], computing=groups[group][2]))
#         shutil.move(tmp_path, path)
#         bar.update(1)
#     else:
#         groups[group][1]+=1
#         bar.set_postfix(dict(prev_done=groups[group][1], computing=groups[group][2]))
#     return lambda : _load(path)

# async def checkpoint_json(func, path: Path, group: str, priority: float):
#     id = str(path)
#     path = base_result_path/path
#     #No need for thread safety, this should always be called from the main thread
#     if not group in groups:
#         groups[group] = [tqdm.tqdm(desc=group, total=0), 0, 0]
#     bar = groups[group][0]
#     if not path.exists():
#         bar.total+=1
#         bar.refresh()
#         tmp_path = path.with_name(".tmp"+path.name)
#         #Its fine if the tmp file lingers, it can be used for bug inspection
#         async with my_limiter.get(priority):
#             groups[group][2]+=1
#             bar.set_postfix(dict(prev_done=groups[group][1], computing=groups[group][2]))
#             def mrun():
#                 res = func()
#                 tmp_path.parent.mkdir(exist_ok=True, parents=True)
#                 with tmp_path.open("w") as f:
#                     json.dump(res, f)
#             try:
#                 await anyio.to_thread.run_sync(mrun)
#             except Exception as e:
#                 raise add_note_to_exception(e, f"While computing {id}")
#             finally:
#                 groups[group][2]-=1
#                 bar.set_postfix(dict(prev_done=groups[group][1], computing=groups[group][2]))
#         shutil.move(tmp_path, path)
#         bar.update(1)
#     else:
#         groups[group][1]+=1
#         bar.set_postfix(dict(prev_done=groups[group][1], computing=groups[group][2]))
#     def load():
#         with path.open("r") as f:
#             return json.load(f)
#     return load


        
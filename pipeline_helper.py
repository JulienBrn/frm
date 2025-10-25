from pathlib import Path
import tqdm.auto as tqdm
import xarray as xr, pandas as pd
import dask.array as da
import anyio, shutil
import heapq, json
from contextlib import asynccontextmanager
from typing import Callable, Awaitable, Any, TypeVar, List, Dict, Union
import logging

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
base_result_path = Path("/media/julienb/T7 Shield/Revue-FRM/AnalysisData/")

def add_note_to_exception(exc, note):
    #Because we are currently using python 3.9 and we dont have add_note
    new_exc = exc.__class__(f"{exc}\nNote: {note}")
    new_exc.__cause__ = exc.__cause__
    new_exc.__context__ = exc.__context__
    return new_exc

T = TypeVar("T")

def mk_checkpoint(
    save_fn: Callable[[T, Path], None],
    load_fn: Callable[[Path], T]
) -> Callable[
    [Callable[[], T], Path, str, float],
    Awaitable[Callable[[], T]]
]:
    async def checkpoint(func, path: Path, group: str, priority: float):
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
                def mrun():
                    res = func()
                    tmp_path.parent.mkdir(exist_ok=True, parents=True)
                    if tmp_path.exists():
                        if tmp_path.is_dir():
                            shutil.rmtree(tmp_path)
                        else:
                            tmp_path.unlink()
                    save_fn(res, tmp_path)
                try:
                    await anyio.to_thread.run_sync(mrun)
                except Exception as e:
                    logger.exception(f"While computing {id}")
                    raise add_note_to_exception(e, f"While computing {id}")
                finally:
                    groups[group][2]-=1
                    bar.set_postfix(dict(prev_done=groups[group][1], computing=groups[group][2]))
            shutil.move(tmp_path, path)
            bar.update(1)
        else:
            groups[group][1]+=1
            bar.set_postfix(dict(prev_done=groups[group][1], computing=groups[group][2]))
        return lambda : load_fn(path)
    return checkpoint

def _load_xarray(path: Path) -> Union[xr.DataArray, xr.Dataset]: 
    obj = xr.open_zarr(path)
    if not obj:
        raise Exception("Load result is None...")
    computed, is_array = obj.attrs.pop("_computed_"), obj.attrs.pop("_is_xrdatarray_")
    if computed:
        obj = obj.compute()
    if is_array:
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
    obj.attrs["_computed_"] = all(not isinstance(v.data, da.Array) for v in obj.variables.values())
    obj.to_zarr(path, compute=True)


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

@asynccontextmanager
async def single(yes=True):
    global lock, do_stop
    if yes:
        async with lock:
            if not do_stop:
                try:
                    yield
                except:
                    do_stop=True
                    raise
            else:
                await anyio.sleep(10000)
    else:
        yield
        

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


        
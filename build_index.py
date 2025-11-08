import os
from pathlib import Path

def build_tree(root_dir):
    """
    Recursively build a nested dict representing the folder structure.
    Only includes .html files.
    """
    tree = {}
    for entry in sorted(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, entry)
        if os.path.isdir(full_path):
            subtree = build_tree(full_path)
            if subtree:  # Only include dirs that contain HTML files
                tree[entry] = subtree
        elif entry.endswith(".html"):
            tree[entry] = None
    return tree


def generate_html(tree, base_path=""):
    """
    Recursively generate nested <ul><li> elements from the folder tree.
    """
    html = "<ul>\n"
    for name, subtree in tree.items():
        if subtree is None:
            rel_path = os.path.join(base_path, name).replace("\\", "/")
            html += f'  <li><a href="{rel_path}">{name[:-5]}</a></li>\n'
        else:
            html += f'  <li><strong>{name}</strong>\n'
            html += generate_html(subtree, os.path.join(base_path, name))
            html += "  </li>\n"
    html += "</ul>\n"
    return html


def generate_index_html(scan_dir, output_file="./index.html"):
    """
    Generate an index.html in the current directory based on HTML files
    found under scan_dir (recursively).
    """
    tree = build_tree(scan_dir)
    html_tree = generate_html(tree, scan_dir)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Figures Index</title>
  <style>
    body {{
      font-family: system-ui, sans-serif;
      margin: 2em;
      background: #fafafa;
      color: #333;
    }}
    h1 {{
      color: #222;
    }}
    ul {{
      list-style: none;
      padding-left: 1em;
      border-left: 2px solid #ddd;
    }}
    li {{
      margin: 0.3em 0;
    }}
    a {{
      text-decoration: none;
      color: #0066cc;
      font-weight: 500;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    strong {{
      color: #444;
    }}
  </style>
</head>
<body>
  <h1>Figures Index</h1>
  {html_tree}
</body>
</html>
"""
    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ Index generated at: {output_path.resolve()}")


if __name__ == "__main__":
    # Folder to scan (change if needed)
    figures_dir = "./figures"
    generate_index_html(figures_dir)

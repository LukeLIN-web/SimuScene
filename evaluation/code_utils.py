import ast
import os
import subprocess
import tempfile
from typing import Set

BANNED_FUNCS = {"namedWindow", "imshow", "waitKey", "destroyAllWindows"}

class Cv2AliasCollector(ast.NodeVisitor):
    def __init__(self):
        self.cv2_aliases: Set[str] = set()   # e.g. {'cv2', 'cv'}
        self.direct_names: Set[str] = set()  # e.g. {'imshow', 'waitKey'}

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if alias.name == "cv2":
                self.cv2_aliases.add(alias.asname or "cv2")

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module == "cv2":
            for alias in node.names:
                name = alias.asname or alias.name
                if alias.name in BANNED_FUNCS:
                    self.direct_names.add(name)

class RemoveGuiCalls(ast.NodeTransformer):
    def __init__(self, cv2_aliases: Set[str], direct_names: Set[str]):
        super().__init__()
        self.cv2_aliases = cv2_aliases or {"cv2"}  
        self.direct_names = direct_names

    def _is_banned_call(self, call: ast.Call) -> bool:
        # 形式1: cv2_alias.imshow(...)
        if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
            base = call.func.value.id
            attr = call.func.attr
            if base in self.cv2_aliases and attr in BANNED_FUNCS:
                return True

        if isinstance(call.func, ast.Name):
            if call.func.id in self.direct_names or call.func.id in BANNED_FUNCS:
                return True

        return False

    def visit_Expr(self, node: ast.Expr):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call) and self._is_banned_call(node.value):
            return ast.Pass()
        return node

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if self._is_banned_call(node):
            return ast.Constant(value=None)
        return node

def sanitize_code(src: str) -> str:
    tree = ast.parse(src)
    collector = Cv2AliasCollector()
    collector.visit(tree)
    transformer = RemoveGuiCalls(collector.cv2_aliases, collector.direct_names)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)  

def run_sanitized(code: str, save_path: str, video_name: str):
    clean_code = sanitize_code(code)

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
        tmp.write(clean_code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["python3", tmp_path, os.path.join(save_path, video_name)],
            capture_output=True,
            text=True,
            check=False,
        )
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        return result.returncode
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

if __name__ == "__main__":
    code = r"""
import cv2 as cv
from cv2 import waitKey

cv.namedWindow('win')           
cv.imshow('win', None)          
x = waitKey(1)                  
cv.destroyAllWindows()          

print('done')
"""
    rc = run_sanitized(code, "/tmp", "demo.mp4")
    print("return code:", rc)

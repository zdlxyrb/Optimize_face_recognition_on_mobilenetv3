import ast
from graphviz import Digraph

class CodeParser(ast.NodeVisitor):
    def __init__(self):
        self.graph = Digraph()

    def visit_ClassDef(self, node):
        # 解析类定义，提取层结构
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                # 处理层定义（如self.conv = nn.Conv2d(...)）
                pass
        self.generic_visit(node)

# 生成架构图
parser = CodeParser()
with open("./innovation_model_v3.py",encoding="utf-8") as f:
    tree = ast.parse(f.read())
parser.visit(tree)
parser.graph.render("model3.png")
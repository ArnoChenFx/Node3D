#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
from Node3D.vendor.NodeGraphQt import BackdropNode
from PySide2 import QtWidgets, QtGui
from Node3D.widgets.mainWindow import mainWindow
from Node3D.nodes.subgraph_nodes import Publish
import inspect
import importlib
import qdarkstyle


def run(nodes=[]):
    app = QtWidgets.QApplication([])
    app.setFont(QtGui.QFont("Consolas", 9))
    win = mainWindow()

    app.setStyleSheet(qdarkstyle.load_stylesheet_from_environment())

    # create node graph.
    graph = win.graph

    # registered nodes.
    BackdropNode.__identifier__ = 'Utility'
    nodes.append(BackdropNode)
    [graph.register_node(n) for n in nodes]

    win.node_tree.update()
    win.show()
    win.closed.connect(lambda: app.exit(0))
    sys.exit(app.exec_())


def get_nodes_from_folder(folder_path):
    if not os.path.exists(folder_path):
        return []
    path, folder_name = os.path.split(folder_path)
    if path not in sys.path:
        sys.path.append(path)

    nodes = []
    for i in os.listdir(folder_path):
        if not i.endswith(".py") or i.startswith("_"):
            continue

        filename = i[:-3]
        module_name = folder_name + "." + filename

        for name, obj in inspect.getmembers(importlib.import_module(module_name)):
            if inspect.isclass(obj) and filename in str(obj):
                if len(inspect.getmembers(obj)) > 0 and obj.__identifier__ != '__None':
                    nodes.append(obj)
    return nodes


def get_published_nodes_from_folder(folder_path):
    if not os.path.exists(folder_path):
        return []
    path, folder_name = os.path.split(folder_path)
    if path not in sys.path:
        sys.path.append(path)

    nodes = []
    for i in os.listdir(folder_path):
        if not i.endswith(".node") and not i.endswith(".json"):
            continue
        file_name = os.path.join(folder_path, i)
        node = Publish.create_node_class(file_name)
        if node is not None:
            nodes.append(node)

    return nodes


if __name__ == '__main__':
    node_path = os.getcwd() + "/Node3D/nodes"
    published_node_path = os.getcwd() + "/Node3D/nodes/published_nodes"
    register_nodes = get_nodes_from_folder(node_path)
    register_nodes.extend(get_published_nodes_from_folder(published_node_path))
    run(register_nodes)
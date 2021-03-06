from .properties import PropFileSavePath
from .. import QtWidgets
import os


class _element_widget(QtWidgets.QWidget):
    def __init__(self):
        super(_element_widget, self).__init__()
        self.layout = QtWidgets.QHBoxLayout(self)


class NodePublishWidget(QtWidgets.QDialog):
    def __init__(self, node):
        super(NodePublishWidget, self).__init__()
        self.setWindowTitle("Publish Node")
        self.published = False

        layout = QtWidgets.QVBoxLayout(self)
        self.node = node
        self.node_name = QtWidgets.QLineEdit(node.name())
        self.node_class_name = QtWidgets.QLineEdit(node.name().replace(" ", "_"))
        self.node_identifier = QtWidgets.QLineEdit('Custom')

        path = os.getcwd() + "/Node3D/nodes/published_nodes"
        if not os.path.exists(path):
            path = os.getcwd()
        path += "/" + self.node_class_name.text() + ".node"
        path = path.replace('\\', '/')

        self.file_path_widget = PropFileSavePath()
        self.file_path_widget.set_file_dir(path)
        self.file_path_widget.set_ext('*.json;*.node')
        self.file_path_widget.set_value(path)

        publish_btn = QtWidgets.QPushButton('Publish')
        publish_btn.pressed.connect(lambda: self.publish())

        cancel_btn = QtWidgets.QPushButton('Cancel')
        cancel_btn.pressed.connect(lambda: self.close())

        name_widget = _element_widget()
        name_widget.layout.addWidget(QtWidgets.QLabel('Node Name'))
        name_widget.layout.addWidget(self.node_name)

        identifier_widget = _element_widget()
        identifier_widget.layout.addWidget(QtWidgets.QLabel('Node Identifier'))
        identifier_widget.layout.addWidget(self.node_identifier)

        class_name_widget = _element_widget()
        class_name_widget.layout.addWidget(QtWidgets.QLabel('Node Class Name'))
        class_name_widget.layout.addWidget(self.node_class_name)

        layout.addWidget(name_widget)
        layout.addWidget(class_name_widget)
        layout.addWidget(identifier_widget)
        layout.addWidget(self.file_path_widget)
        layout.addWidget(publish_btn)
        layout.addWidget(cancel_btn)

        layout.addStretch()

    def publish(self):
        if self.published:
            return
        file_path = self.file_path_widget.get_value()
        node_name = self.node_name.text()
        node_identifier = self.node_identifier.text()
        node_class_name = self.node_class_name.text()
        self.node.publish(file_path, node_name, node_identifier, node_class_name)
        self.published = True
        self.close()


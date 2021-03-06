from Node3D.base.node import GeometryNode
import numpy as np


class GetAttributeData(GeometryNode):
    __identifier__ = 'Attribute'
    NODE_NAME = 'Get_Attribute_Data'

    def __init__(self):
        super(GetAttributeData, self).__init__(False)
        self.create_property('out data', None)
        self.add_input('geo', GeometryNode)
        self.add_output('out data', np.ndarray)
        self.set_parameters([{'name': 'Attribute Class', 'type': 'list', 'value': 'vertex',
                              'limits': ['vertex', 'edge', 'face', 'detail']},
                             {'name': 'Attribute Name', 'type': 'listText'}])

    def get_data(self, port):
        try:
            return self.get_property('out data')
        except:
            return None

    def run(self):
        geo = self.get_input_geometry_ref(0)
        if geo is None:
            self.set_property('out data', None)
            self.error('Please input a geometry')
            return

        attrib_class = self.get_property('Attribute Class')
        self.update_list_text_param('Attribute Name', geo.getAttribNames()[attrib_class])

        attrib_name = self.get_property('Attribute Name')
        if not geo.hasAttribute(attrib_class, attrib_name):
            self.set_property('out data', None)
            return

        if attrib_class == 'vertex':
            data = geo.getVertexAttribData(attrib_name)
        elif attrib_class == 'edge':
            data = geo.getEdgeAttribData(attrib_name)
        elif attrib_class == 'face':
            data = geo.getFaceAttribData(attrib_name)
        elif attrib_class == 'detail':
            data = geo.getDetailAttrib(attrib_name)

        self.set_property('out data', data)


class SetAttributeData(GeometryNode):
    __identifier__ = 'Attribute'
    NODE_NAME = 'Set_Attribute_Data'

    def __init__(self):
        super(SetAttributeData, self).__init__()
        self.create_property('out data', None)
        self.add_input('geo', GeometryNode)
        self.add_input('in data', np.ndarray)

        self.set_parameters([{'name': 'Attribute Class', 'type': 'list', 'value': 'vertex',
                              'limits': ['vertex', 'edge', 'face', 'detail']},
                             {'name': 'Attribute Name', 'type': 'listText'}])

    def run(self):
        if not self.copyData():
            return

        attrib_class = self.get_property('Attribute Class')
        self.update_list_text_param('Attribute Name', self.geo.getAttribNames()[attrib_class])

        attrib_name = self.get_property('Attribute Name')
        if not self.geo.hasAttribute(attrib_class, attrib_name):
            return

        data = self.get_input_data(1)
        if data is None:
            return

        if attrib_class == 'detail':
            self.geo.setDetailAttrib(attrib_name, data)
            return

        if attrib_class == 'vertex':
            if len(data) != self.geo.getNumVertexes():
                self.error('data size not match')
                return
            self.geo.setVertexAttribData(attrib_name, data)
        elif attrib_class == 'edge':
            if len(data) != self.geo.getNumEdges():
                self.error('data size not match')
                return
            self.geo.setEdgeAttribData(attrib_name, data)
        elif attrib_class == 'face':
            if len(data) != self.geo.getNumFaces():
                self.error('data size not match')
                return
            self.geo.setFaceAttribData(attrib_name, data)


class AttributeDelete(GeometryNode):
    __identifier__ = 'Attribute'
    NODE_NAME = 'Attribute_Delete'

    def __init__(self):
        super(AttributeDelete, self).__init__()
        self.set_parameters([{'name': 'Attribute Class', 'type': 'list', 'value': 'vertex',
                              'limits': ['vertex', 'edge', 'face', 'detail']},
                             {'name': 'Attribute Name', 'type': 'listText'}])
        self.add_input('geo', GeometryNode)

    def run(self):
        if not self.copyData():
            return

        attrib_class = self.get_property('Attribute Class')
        self.update_list_text_param('Attribute Name', self.geo.getAttribNames()[attrib_class])
        attrib_name = self.get_property('Attribute Name')
        self.geo.removeAttribute(attrib_class, attrib_name)


class AttributeCreate(GeometryNode):
    __identifier__ = 'Attribute'
    NODE_NAME = 'Attribute_Create'

    def __init__(self):
        super(AttributeCreate, self).__init__()
        self.set_parameters([{'name': 'Attribute Class', 'type': 'list', 'value': 'vertex',
                              'limits': ['vertex', 'edge', 'face', 'detail']},
                             {'name': 'Attribute Name', 'type': 'str'},
                             {'name': 'Attribute Type', 'type': 'list', 'value': 'float',
                              'limits': ['float', 'int', 'vector2', 'vector3', 'vector4', 'matrix3', 'matrix4', 'bool',
                                         'list', 'tuple', 'custom', 'str']},
                             {'name': 'Default Value', 'type': 'str', 'value': '0.0'},
                             {'name': 'Attribute Value', 'type': 'str', 'value': '0.0'}])
        self.add_input('geo', GeometryNode)

    def get_value(self, value, attrib_type):
        if attrib_type == 'float':
            value = float(value)
        elif attrib_type == 'int':
            value = int(value)
        elif attrib_type == 'bool':
            value = bool(value)
        elif attrib_type == 'str':
            value = str(value)
        elif attrib_type == 'tuple':
            value = tuple(value)
        elif attrib_type == 'list':
            value = list(value)
        elif 'vector' in attrib_type:
            if type(value) is not list:
                self.error('please input a list value')
                return
            if attrib_type == 'vector3':
                if len(value) != 3:
                    self.error('please input a list with 3 values')
                    return
            elif attrib_type == 'vector2':
                if len(value) != 2:
                    self.error('please input a list with 2 values')
                    return
            elif attrib_type == 'vector4':
                if len(value) != 4:
                    self.error('please input a list with 4 values')
                    return
            else:
                return
        elif attrib_type == 'matrix3':
            if type(value) is list:
                if len(value) != 9:
                    self.error('please input a list with 9 values or input 0/1')
                    return
                value = np.array(value).reshape(3, 3)
            else:
                if value == 0:
                    value = np.zeros((3, 3))
                else:
                    value = np.identity(3)
        elif attrib_type == 'matrix4':
            if type(value) is list:
                if len(value) != 16:
                    self.error('please input a list with 16 values or input 0/1')
                    return
                value = np.array(value).reshape(4, 4)
            else:
                if value == 0:
                    value = np.zeros((4, 4))
                else:
                    value = np.identity(4)
        return value

    def run(self):
        if not self.copyData():
            return
        attrib_name = self.get_property('Attribute Name')
        if not attrib_name:
            return

        attrib_type = self.get_property('Attribute Type')

        try:
            value = self.get_property('Attribute Value')
            default_value = self.get_property('Default Value')
            if attrib_type != 'string':
                value = eval(value)
                default_value = eval(default_value)
            value = self.get_value(value, attrib_type)
            default_value = self.get_value(default_value, attrib_type)
        except:
            self.error('please input a valid value')
            return

        attrib_class = self.get_property('Attribute Class')

        self.geo.createAttribute(attrib_class, attrib_name, attribType=attrib_type, defaultValue=value, applyValue=True)
        self.geo.attributeMap[attrib_class][attrib_name]['default_value'] = default_value


class AttributePromote(GeometryNode):
    __identifier__ = 'Attribute'
    NODE_NAME = 'Attribute_Promote'

    def __init__(self):
        super(AttributePromote, self).__init__()
        self.set_parameters(
            [{'name': 'From', 'type': 'list', 'value': 'vertex', 'limits': ['vertex', 'edge', 'face', 'detail']},
             {'name': 'To', 'type': 'list', 'value': 'vertex', 'limits': ['vertex', 'edge', 'face', 'detail']},
             {'name': 'Attribute Name', 'type': 'listText'},
             {'name': 'New Attribute Name', 'type': 'str'},
             {'name': 'Delete Origin', 'type': 'bool', 'value': True}])

        self.add_input('geo', GeometryNode)

    def run(self):
        if not self.copyData():
            return

        from_class = self.get_property('From')
        to_class = self.get_property('To')

        self.update_list_text_param('Attribute Name', self.geo.getAttribNames()[from_class])
        attrib_name = self.get_property('Attribute Name')
        if not self.geo.hasAttribute(from_class, attrib_name):
            return

        new_name = self.get_property('New Attribute Name')

        if not new_name:
            return

        if from_class == 'vertex':
            if to_class == 'face':
                self.geo.meshFuncs.VertexToFace(attrib_name, new_name)
            elif to_class == 'edge':
                self.geo.meshFuncs.VertexToEdge(attrib_name, new_name)
            elif to_class == 'vertex':
                if attrib_name != new_name:
                    self.geo.copyAttribute('vertex', attrib_name, new_name)
        elif from_class == 'face':
            if to_class == 'vertex':
                self.geo.meshFuncs.FaceToVertex(attrib_name, new_name)
            elif to_class == 'face':
                if attrib_name != new_name:
                    self.geo.copyAttribute('face', attrib_name, new_name)
        elif from_class == 'edge':
            if to_class == 'vertex':
                self.geo.meshFuncs.EdgeToVertex(attrib_name, new_name)
            elif to_class == 'edge':
                if attrib_name != new_name:
                    self.geo.copyAttribute('edge', attrib_name, new_name)

        if self.get_property('Delete Origin'):
            self.geo.removeAttribute(from_class, attrib_name)


class AttributeRename(GeometryNode):
    __identifier__ = 'Attribute'
    NODE_NAME = 'Attribute_Rename'

    def __init__(self):
        super(AttributeRename, self).__init__()
        self.set_parameters([{'name': 'Attribute Class', 'type': 'list', 'value': 'vertex',
                              'limits': ['vertex', 'edge', 'face', 'detail']},
                             {'name': 'Attribute Name', 'type': 'listText'},
                             {'name': 'New Attribute Name', 'type': 'str'}])
        self.add_input('geo', GeometryNode)

    def run(self):
        if not self.copyData():
            return

        attrib_class = self.get_property('Attribute Class')
        self.update_list_text_param('Attribute Name', self.geo.getAttribNames()[attrib_class])
        attrib_name = self.get_property('Attribute Name')
        if not self.geo.hasAttribute(attrib_class, attrib_name):
            return

        new_name = self.get_property('New Attribute Name')

        if not new_name:
            return

        self.geo.renameAttribute(attrib_class, attrib_name, new_name)

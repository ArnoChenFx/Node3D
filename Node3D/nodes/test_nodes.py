from Node3D.base.node import AutoNode


def print100(node):
    print(100)
    print(node.name())


params = [
    {'name': 'Basic parameter data types', 'type': 'group', 'children': [
        {'name': 'Integer', 'type': 'int', 'value': 10},
        {'name': 'Float', 'type': 'float', 'value': 10.5},
        {'name': 'String', 'type': 'str', 'value': "hi"},
        {'name': 'List', 'type': 'list', 'values': [1, 2, 3], 'value': 1},
        {'name': 'Named List', 'type': 'list', 'values': {"one": 1, "two": "twosies", "three": [3, 3, 3]}, 'value': 2},
        {'name': 'Boolean', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
        {'name': 'Color', 'type': 'color', 'value': (100,255,100,255), 'tip': "This is a color button"},
        {'name': 'Gradient', 'type': 'colorRamp'},
        {'name': 'Subgroup', 'type': 'group', 'children': [
            {'name': 'Sub-param 1', 'type': 'int', 'value': 10},
            {'name': 'Sub-param 2', 'type': 'float', 'value': 1.2e6},
        ]},
        {'name': 'Text Parameter', 'type': 'text', 'value': 'Some text...'},
        {'name': 'Action Parameter', 'type': 'action'},
    ]},
    {'name': 'Numerical Parameter Options', 'type': 'group', 'children': [
        {'name': 'Units + SI prefix', 'type': 'float_', 'value': 1.2},
        {'name': 'Limits (min=7;max=15)', 'type': 'int_', 'value': 11, 'limits': (7, 15), 'default': -6},
        {'name': 'DEC stepping', 'type': 'float', 'value': 1.2e6},

    ]},
    {'name': 'Save/Restore functionality', 'type': 'group', 'children': [
        {'name': 'Save State', 'type': 'action'},
        {'name': 'Restore State', 'type': 'action', 'children': [
            {'name': 'Add missing items', 'type': 'bool', 'value': True},
            {'name': 'Remove extra items', 'type': 'bool', 'value': True},
        ]},
    ]},
    {'name': 'Extra Parameter Options', 'type': 'group', 'children': [
        {'name': 'Read-only', 'type': 'float', 'value': 1.2e6,  'readonly': True},
        {'name': 'Renamable', 'type': 'float', 'value': 1.2e6,  'renamable': True},
        {'name': 'Removable', 'type': 'float', 'value': 1.2e6,  'removable': True},
    ]},
    {'name': 'ARNO', 'type': 'group', 'children': [
        {'name': 'FilePath', 'type': 'file', 'value': 'xxx', 'ext': '*;;*.obj'},
        {'name': 'vec', 'type': 'vector3', 'value': [0, 1, 2]},
        {'name': 'Gradient', 'type': 'colorRamp'},
        {'name': 'Curve', 'type': 'curveRamp'},
        {'name': 'Click me', 'type': 'action', "funcs": [print100]},
    ]},

]


class TestNode(AutoNode):

    # set a unique node identifier.
    __identifier__ = 'Test'

    # set the initial default node name.
    NODE_NAME = 'Test Params'

    def __init__(self):
        super(TestNode, self).__init__(float, float)

        self.set_parameters(params)
        self.add_input("in int", int)
        self.add_input("in vector", list)
        self.add_output("out")

        self.create_property("aaa", 10)
        self.create_property('bbb', 20)

    def getData(self, port):
        return self.get_property('aaa') * self.get_property('bbb')

    def run(self):
        print("run: ", self.name())

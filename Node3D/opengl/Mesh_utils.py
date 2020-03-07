import numpy as np
import openmesh as om
import copy
import numba
from numba import cuda
from Qt import QtCore, QtGui
from ..vendor.pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from OpenGL.GL import *
from ..vendor.pyqtgraph import functions as fn
import copy

@numba.jit(nopython=True, cache=True, nogil=True)  # parallel=True, nogil=True
def _calVertexSmooth(data, iter, vv):
    for i in range(iter):
        for v in range(vv.shape[0]):
            pos = data[v]
            num = 1
            for ov in vv[v]:
                if ov >= 0:
                    pos += data[ov]
                    num += 1
            if num > 1:
                data[v] = pos / num


class MeshFuncs(object):
    def __init__(self, geo):
        self.geo = geo
        self.mesh = geo._mesh

    def FaceToVertex(self, name, newName=None):
        if not self.geo.hasAttribute("face", name):
            return

        is_array = self.geo.getAttribIsArray('face', name)

        if newName is None:
            newName = name

        attr = self.geo.getFaceAttribData(name)
        self.geo.setAttribInfo('vertex', newName, self.geo.getAttribInfo('face', name))

        if not is_array:
            for v in self.mesh.vertices():
                for f in self.mesh.vf(v):
                    self.mesh.set_vertex_property(newName, v, attr[f.idx()])
                    break
        else:
            for v in self.mesh.vertices():
                res = None
                num = 0.0
                for f in self.mesh.vf(v):
                    at = attr[f.idx()]
                    if res is None:
                        res = copy.deepcopy(at)
                    else:
                        res += at
                    num += 1.0

                if res is not None:
                    self.mesh.set_vertex_property(newName, v, res / num)
        self.geo.signals.emit_attribChanged()

    def VertexToFace(self, name, newName=None):
        if not self.geo.hasAttribute("vertex", name):
            return

        is_array = self.geo.getAttribIsArray('vertex', name)

        if newName is None:
            newName = name

        attr = self.geo.getVertexAttribData(name)
        self.geo.setAttribInfo('face', newName, self.geo.getAttribInfo('vertex', name))

        if not is_array:
            for f in self.mesh.faces():
                for v in self.mesh.fv(f):
                    self.mesh.set_face_property(newName, f, attr[v.idx()])
                    break
        else:
            for f in self.mesh.faces():
                res = None
                num = 0.0
                for v in self.mesh.fv(f):
                    at = attr[v.idx()]
                    if res is None:
                        res = copy.deepcopy(at)
                    else:
                        res += at
                    num += 1.0

                if res is not None:
                    self.mesh.set_face_property(newName, f, res / num)
        self.geo.signals.emit_attribChanged()

    def EdgeToVertex(self, name, newName=None):
        if not self.geo.hasAttribute("edge", name):
            return

        is_array = self.geo.getAttribIsArray('edge', name)

        if newName is None:
            newName = name

        attr = self.geo.getEdgeAttribData(name)
        self.geo.setAttribInfo('vertex', newName, self.geo.getAttribInfo('edge', name))

        if not is_array:
            for v in self.mesh.vertices():
                for e in self.mesh.ve(v):
                    self.mesh.set_vertex_property(newName, v, attr[e.idx()])
                    break

        else:
            for v in self.mesh.vertices():
                res = None
                num = 0.0
                for e in self.mesh.ve(v):
                    at = attr[e.idx()]
                    if res is None:
                        res = copy.deepcopy(at)
                    else:
                        res += at
                    num += 1.0

                if res is not None:
                    self.mesh.set_vertex_property(newName, v, res / num)
        self.geo.signals.emit_attribChanged()

    def VertexToEdge(self, name, newName=None):
        if not self.geo.hasAttribute("vertex", name):
            return

        is_array = self.geo.getAttribIsArray('vertex', name)

        if newName is None:
            newName = name

        attr = self.geo.getVertexAttribData(name)
        evs = self.mesh.ev_indices()
        self.geo.setAttribInfo('edge', newName, self.geo.getAttribInfo('vertex', name))

        if not is_array:
            for _e, _vts in enumerate(evs):
                e = self.mesh.edge_handle(_e)
                v = self.mesh.vertex_handle(_vts[0])
                self.mesh.set_edge_property(newName, e, attr[v.idx()])
        else:
            for _e, _vts in enumerate(evs):
                e = self.mesh.edge_handle(_e)
                res = None
                num = 0.0
                for _v in _vts:
                    at = attr[self.mesh.vertex_handle(_v).idx()]
                    if res is None:
                        res = copy.deepcopy(at)
                    else:
                        res += at
                    num += 1.0
                if res is not None:
                    self.mesh.set_edge_property(newName, e, res / num)

        self.geo.signals.emit_attribChanged()

    def flipEdges(self, edges):
        for e in edges:
            if not self.mesh.is_boundary(e):
                self.mesh.flip(e)

        self.geo.signals.emit_attribChanged()

    def collapsingEdges(self, from_vertex, to_vertex):
        fromv = self.mesh.vertex_handle(from_vertex)
        tov = self.mesh.vertex_handle(to_vertex)

        he = self.mesh.find_halfedge(fromv, tov)
        if he is not None and he.idx() >= 0:
            self.mesh.collapse(he)
            self.geo.signals.emit_attribChanged()

    def smoothMesh(self, iter=1):
        data = self.mesh.points()
        _calVertexSmooth(data, iter, self.mesh.vertex_vertex_indices())

    def smoothVertexAttrib(self, name, iter=1):
        if not self.geo.getAttribIsArray('vertex', name):
            return
        data = self.geo.getVertexAttribData(name)
        if data is None:
            return
        _calVertexSmooth(data, iter, self.mesh.vertex_vertex_indices())
        self.geo.setVertexAttribData(name, data, True)

    def facePos(self, apply=False):
        result = np.zeros((self.mesh.n_faces(), 3), dtype=np.float64)
        for f in self.mesh.faces():
            result[f.idx()] = self.mesh.calc_face_centroid(f)

        if apply:
            self.geo.setFaceAttribData("pos", result, attribType='vector3', defaultValue=[0, 0, 0])
        else:
            return result

    def edgeLength(self, apply=False):
        result = np.zeros((self.mesh.n_edges(),), dtype=np.float32)
        for e in self.mesh.edges():
            result[e.idx()] = self.mesh.calc_edge_length(e)

        if apply:
            self.geo.setEdgeAttribData("length", result, attribType='float', defaultValue=0.0)
        else:
            return result

    @staticmethod
    def applyMatrix(data, mat, offset=None):
        sp = data.shape[1]
        if sp < 3 or sp > 4:
            return

        if type(mat) is np.ndarray or type(mat) is np.matrix:
            matrix = mat
            dim = mat.shape[1]
        else:
            return

        if dim == 4:
            if sp == 3:
                positions = np.c_[data, np.ones(data.shape[0])]
            else:
                positions = data
        elif dim == 3:
            if sp == 3:
                positions = data
            else:
                positions = data[..., [0, 1, 2]]
        else:
            return

        res = positions.dot(matrix)
        if sp == 3 or dim == 3:
            data[..., [0, 1, 2]] = res[..., [0, 1, 2]]
        else:
            data[..., [0, 1, 2, 3]] = res
        if offset is not None:
            data[..., [0, 1, 2]] += offset

    def transformVertex(self, mat, offset=None):
        points = self.mesh.points()

        if type(mat) is np.ndarray or type(mat) is np.matrix:
            matrix = mat
            if mat.shape[1] == 3:
                positions = points
            else:
                positions = np.c_[points, np.ones(points.shape[0])]
        else:
            return

        res = positions.dot(matrix)
        points[..., [0, 1, 2]] = res[..., [0, 1, 2]]

        if offset is not None:
            points[..., [0, 1, 2]] += offset


class mesh_signals(QtCore.QObject):
    attribChanged = QtCore.Signal()

    def __init__(self):
        super().__init__()

    def emit_attribChanged(self):
        self.attribChanged.emit()


class bbox(GLGraphicsItem):
    def __init__(self):
        super(bbox, self).__init__()
        self.size = [1, 1, 1]
        self.setColor((255, 255, 0, 200))
        self.setGLOptions('translucent')

    def set(self, pos, size):
        pos[0] -= size[0] * 0.5
        pos[1] -= size[1] * 0.5
        pos[2] -= size[2] * 0.5

        self.size = size
        tr = QtGui.QMatrix4x4()
        tr.translate(pos[0], pos[1], pos[2])
        return tr

    def setColor(self, *args):
        self.__color = fn.Color(*args)

    def color(self):
        return self.__color

    def paint(self):
        # self.setupGLState()

        glBegin(GL_LINES)

        glColor4f(*self.color().glColor())
        x, y, z = self.size
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, z)
        glVertex3f(x, 0, 0)
        glVertex3f(x, 0, z)
        glVertex3f(0, y, 0)
        glVertex3f(0, y, z)
        glVertex3f(x, y, 0)
        glVertex3f(x, y, z)

        glVertex3f(0, 0, 0)
        glVertex3f(0, y, 0)
        glVertex3f(x, 0, 0)
        glVertex3f(x, y, 0)
        glVertex3f(0, 0, z)
        glVertex3f(0, y, z)
        glVertex3f(x, 0, z)
        glVertex3f(x, y, z)

        glVertex3f(0, 0, 0)
        glVertex3f(x, 0, 0)
        glVertex3f(0, y, 0)
        glVertex3f(x, y, 0)
        glVertex3f(0, 0, z)
        glVertex3f(x, 0, z)
        glVertex3f(0, y, z)
        glVertex3f(x, y, z)

        glEnd()

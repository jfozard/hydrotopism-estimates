# -*- python -*-
#
#       OpenAlea.Core
#
#       Copyright 2006-2009 INRIA - CIRAD - INRA
#
#       File author(s): Jerome Chopard <jerome.chopard@sophia.inria.fr>
#                       Fred Theveny <frederic.theveny@cirad.fr>
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       OpenAlea WebSite: http://openalea.gforge.inria.fr
#
################################################################################
"""This module provide a set of concepts to add properties to graph elements"""

__license__ = "Cecill-C"
__revision__ = " $Id: property_graph.py 11356 2011-11-16 17:44:53Z pradal $ "

from interface.property_graph import IPropertyGraph, PropertyError
from graph import Graph, InvalidVertex, InvalidEdge

class PropertyGraph(IPropertyGraph, Graph):
    """
    Simple implementation of IPropertyGraph using
    dict as properties and two dictionaries to
    maintain these properties
    """
    def __init__(self, graph=None, **kwds):
        self._vertex_property = {}
        self._edge_property = {}
        self._graph_property = {}
        Graph.__init__(self, graph, **kwds)

    def vertex_property_names(self):
        """todo"""
        return self._vertex_property.iterkeys()
    vertex_property_names.__doc__ = IPropertyGraph.vertex_property_names.__doc__

    def vertex_property(self, property_name):
        """todo"""
        try:
            return self._vertex_property[property_name]
        except KeyError:
            raise PropertyError("property %s is undefined on vertices"
                                % property_name)
    vertex_property.__doc__=IPropertyGraph.vertex_property.__doc__

    def edge_property_names(self):
        """todo"""
        return self._edge_property.iterkeys()
    edge_property_names.__doc__ = IPropertyGraph.edge_property_names.__doc__

    def edge_property(self, property_name):
        """todo"""
        try:
            return self._edge_property[property_name]
        except KeyError:
            raise PropertyError("property %s is undefined on edges"
                                % property_name)
    edge_property.__doc__ = IPropertyGraph.edge_property.__doc__

    def graph_property(self):
        """todo"""
        return self._graph_property
    graph_property.__doc__ = IPropertyGraph.graph_property.__doc__

    def add_vertex_property(self, property_name):
        """todo"""
        if property_name in self._vertex_property:
            raise PropertyError("property %s is already defined on vertices"
                                % property_name)
        self._vertex_property[property_name] = {}
    add_vertex_property.__doc__ = IPropertyGraph.add_vertex_property.__doc__

    def remove_vertex_property(self, property_name):
        """todo"""
        try:
            del self._vertex_property[property_name]
        except KeyError:
            raise PropertyError("property %s is undefined on vertices"
                                % property_name)
    remove_vertex_property.__doc__ = IPropertyGraph.remove_vertex_property.__doc__

    def add_edge_property(self, property_name):
        """todo"""
        if property_name in self._edge_property:
            raise PropertyError("property %s is already defined on edges"
                                % property_name)
        self._edge_property[property_name] = {}
    add_edge_property.__doc__ = IPropertyGraph.add_edge_property.__doc__

    def remove_edge_property(self, property_name):
        """todo"""
        try:
            del self._edge_property[property_name]
        except KeyError:
            raise PropertyError("property %s is undefined on edges"
                                % property_name)
    remove_edge_property.__doc__ = IPropertyGraph.remove_edge_property.__doc__

    def remove_vertex(self, vid):
        """todo"""
        for prop in self._vertex_property.itervalues():
            prop.pop(vid, None)
        Graph.remove_vertex(self, vid)
    remove_vertex.__doc__ = Graph.remove_vertex.__doc__

    def clear(self):
        """todo"""
        for prop in self._vertex_property.itervalues():
            prop.clear()
        for prop in self._edge_property.itervalues():
            prop.clear()
        Graph.clear(self)
    clear.__doc__ = Graph.clear.__doc__

    def remove_edge(self, eid):
        """todo"""
        for prop in self._edge_property.itervalues():
            prop.pop(eid, None)
        Graph.remove_edge(self, eid)
    remove_edge.__doc__ = Graph.remove_edge.__doc__

    def clear_edges(self):
        """todo"""
        for prop in self._edge_property.itervalues():
            prop.clear()
        Graph.clear_edges(self)
    clear_edges.__doc__ = Graph.clear_edges.__doc__

    def extend(self, graph):
        """todo"""
        trans_vid, trans_eid = Graph.extend(self,graph)
        # update properties on vertices
        for prop_name in graph.vertex_property_names():
            if prop_name not in self._vertex_property:
                self.add_vertex_property(prop_name)
            prop = self.vertex_property(prop_name)

            for vid, val in graph.vertex_property(prop_name).iteritems():
                prop[trans_vid[vid]] = val
        # update properties on edges
        for prop_name in graph.edge_property_names():
            if prop_name not in self._edge_property:
                self.add_edge_property(prop_name)
            prop = self.edge_property(prop_name)

            for eid, val in graph.edge_property(prop_name).iteritems():
                prop[trans_eid[eid]] = val

        # update properties on graph
        prop = self.graph_property()
        prop.update(graph.graph_property())

        return trans_vid, trans_eid
    extend.__doc__ = Graph.extend.__doc__


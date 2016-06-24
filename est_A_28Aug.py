
import sys
import numpy as np
from openalea2.celltissue import TissueDB
from openalea2.tissueshape import edge_length
from matplotlib.lines import Line2D
import matplotlib.pylab as plt
from collections import defaultdict
import scipy.linalg as la

ct_name_map = { 'Epidermis': 20, 
                'Cortex': 14, 
                'Endodermis': 6, 
                'Pericycle': 22, 
                'Vasculature': 11,
                'Primordium': 1 }

ct_order = [ 20, 14, 6, 22, 11 ]

# Radial, Inner, Outer cell walls

ct_thick = { 20:[86., 93., 198.], 
             14:[77., 80., 105.],
             6:[45., 48., 70.],
             22:[38., 0., 47.],
             11:[0., 0., 0.],
             0:[0., 0., 0.]}

def classify_wall(ct1, ct2):
    if ct1==ct2:
        return 0
    if ct2==0:
        return 2
    if ct_order.index(ct1) < ct_order.index(ct2):
        return 1
    return 2

def pairs(lst):
    """Generaterator which returns all the sequential pairs of lst,
    treating lst as a circular list
    
    >>> list(pairs([1,2,3]))
    [(1, 2), (2, 3), (3, 1)]
    """     

    i = iter(lst)
    first = prev = i.next()
    for item in i:
        yield prev, item
        prev = item
    yield item, first

def area(poly):
    """ 
    Calculates the area of a polygon
    
    Args:
        poly: list of the vertices of a polygon in counterclockwise order

    Returns:
        float. Area of the polygon
    """
    return sum((0.5*(pp[0][0]*pp[1][1]-pp[1][0]*pp[0][1]) 
                for pp in pairs(poly)))


def ordered_wids_pids(mesh, pos, cid):
    wid_set = set(mesh.borders(2,cid))
    wid = wid_set.pop()
    o_wids = [wid]
    o_pids = list(mesh.borders(1, wid))
    end = o_pids[-1]
    while wid_set:
        for wid in wid_set:
            if end in mesh.borders(1, wid):
                wid_set.discard(wid)
                o_wids.append(wid)
                end = (set(mesh.borders(1, wid)) - set([end])).pop()
                o_pids.append(end)
                break
    o_pids.pop()
    if area([pos[pid] for pid in o_pids])<0:
        o_wids.reverse()
        o_pids=[o_pids[0]]+o_pids[-1:0:-1]
    return o_wids, o_pids

def rot(v):
    return np.array((v[1], -v[0]))

def centroid(poly):
    """ 
    Calculates the centroid of a polygon, defined by
    a list of points

    Args:
        poly: list of the vertices of a polygon in counterclockwise order

    Returns:
        numpy.ndarray. Centroid of the polygon
    
    >>> centroid([(0,0),(1,0),(1,1),(0,1)])
    array([ 0.5,  0.5])
    """
    A = area(poly)
    cx = sum(((pp[0][0]+pp[1][0])*(pp[0][0]*pp[1][1]-pp[1][0]*pp[0][1]) 
            for pp in pairs(poly)))/(6.0*A)
    cy = sum(((pp[0][1]+pp[1][1])*(pp[0][0]*pp[1][1]-pp[1][0]*pp[0][1]) 
            for pp in pairs(poly)))/(6.0*A)
    return np.array((cx, cy))


from lxml import etree
import copy, os

def plot_SVG(db, filename):
    svg = etree.Element("svg", id='svg2', xmlns='http://www.w3.org/2000/svg', version='1.0')    
    t = db.tissue()
    cfg = db.get_config('config')
    scale = 0.02
    
    mesh = t.relation(cfg.mesh_id)
    pos = db.get_property('position')
    cell_type = db.get_property('cell_type')
    
    min_x = min(p[0] for p in pos.itervalues())
    min_y = min(p[1] for p in pos.itervalues())
    max_x = max(p[0] for p in pos.itervalues())
    max_y = max(p[1] for p in pos.itervalues())

    length = max_x-min_x
    width = max_y-min_y

    drawing = etree.SubElement(svg, 'g', transform='matrix(1,0,0,-1,20,%f)'%(20+max_y))
    tissue = etree.SubElement(drawing, 'g')
    cells = etree.SubElement(drawing, 'g', style='fill:none; stroke: blue')
    """
    for cid in list(mesh.wisps(2)):
        wids, pids = ordered_wids_pids(mesh, pos, cid)
        pts = [(pos[pid][0], pos[pid][1]) for pid in pids]
        path = 'M %f,%f '%tuple(pts[0]) + ' '.join('%f,%f'%tuple(x) for x in pts[1:]) +' Z'
        etree.SubElement(cells, 'path', d=path)            
    """
    lPIN = etree.SubElement(tissue, 'g', style='fill:blue; stroke:none')
    lPIN2 = etree.SubElement(tissue, 'g', style='fill:red; stroke:none')
    for cid in mesh.wisps(2):
        wids, pids = ordered_wids_pids(mesh, pos, cid)
        poly = [pos[pid] for pid in pids]
        a = area(poly)
        c = centroid(poly)
        ct = cell_type[cid]
        for wid, (pid1, pid2) in zip(wids, pairs(pids)):
            p1 = pos[pid1]
            p2 = pos[pid2]
            s = set(mesh.regions(1, wid)) - set([cid])
            if s:
                o_cid = s.pop()
                o_ct = cell_type[o_cid]
            else:
                o_ct = 0
            i = classify_wall(ct, o_ct)
            w = ct_thick[ct][i]*scale
            n = rot(p1-p2)
            n = n / la.norm(n)
            d1 = p1+w/np.dot(n, (c-p1))*(c-p1)
            d2 = p2+w/np.dot(n, (c-p2))*(c-p2)
            path = 'M %f,%f '%tuple(p1) + '%f,%f '%tuple(p2) + '%f,%f '%tuple(d2) + '%f, %f '%tuple(d1) + 'Z'
            if ct==14 and c[0]>0:
                etree.SubElement(lPIN2, 'path', d=path)
            else:
                etree.SubElement(lPIN, 'path', d=path)
    etree.ElementTree(svg).write(filename, xml_declaration=True, encoding='utf-8')





def main():
    db = TissueDB()
    db.read("test2.zip")
    cell_type = db.get_property('cell_type')
    t = db.tissue()
    cfg = db.get_config('config')
    
    mesh = t.relation(cfg.mesh_id)

#    print cfg.cell_types

    pos = db.get_property('position')

    box_max = np.max(np.array(list(pos.itervalues())), axis=0)
    box_min = np.min(np.array(list(pos.itervalues())), axis=0)
    print box_max, box_min
    mid = 0.5*(box_max+box_min)
    scale = 120.0/(box_max-box_min)
    scale = 0.5*(scale[0]+scale[1])

    for pid in pos:
        pos[pid] = scale*(pos[pid]-mid)

    centroids = {}
    area_tot = 0.0
    c_tot = np.zeros((2,))
    for cid in mesh.wisps(2):
        _, o_pids = ordered_wids_pids(mesh, pos, cid)
        poly = [pos[pid] for pid in o_pids]
        a = area(poly)
        c = centroid(poly)
        centroids[cid] = c
#        print a, c, c_tot
        c_tot += a*c
        area_tot += a
    c_tot = c_tot/area_tot
    for pid in pos:
        pos[pid] -= c_tot
    for cid in centroids:
        centroids[cid] -= c_tot
        
    box_max = np.max(np.array(list(pos.itervalues())), axis=0)
    box_min = np.min(np.array(list(pos.itervalues())), axis=0)

    
    A_w = 0.0
    wall_thickness = defaultdict(float)
    for cid in mesh.wisps(2):
        ct = cell_type[cid]
        for wid in mesh.borders(2, cid):
            s = set(mesh.regions(1, wid)) - set([cid])
            if s:
                o_cid = s.pop()
                o_ct = cell_type[o_cid]
            else:
                o_ct = 0
            i = classify_wall(ct, o_ct)
            h = ct_thick[ct][i]
            A_w += h*edge_length(mesh, pos, wid)
            wall_thickness[wid] = wall_thickness[wid] + h


    print 'A_w', A_w


    h_cw = 0.0
    for cid in mesh.wisps(2):
        ct = cell_type[cid]
        if ct!=14 or centroids[cid][0]<0:
            continue
        for wid in mesh.borders(2, cid):
            s = set(mesh.regions(1, wid)) - set([cid])
            if s:
                o_cid = s.pop()
                o_ct = cell_type[o_cid]
            else:
                o_ct = 0
            i = classify_wall(ct, o_ct)
            h = ct_thick[ct][i]
            h_cw += h*edge_length(mesh, pos, wid)

    print 'h_cw, h_cw/A_w', h_cw, h_cw/A_w



    h_eta_cw = np.array((0.0, 0.0))
    h_eta2_cw = 0.0
    h_eta2 = 0.0
    for cid in mesh.wisps(2):
        ct = cell_type[cid]
        for wid in mesh.borders(2, cid):
            s = set(mesh.regions(1, wid)) - set([cid])
            if s:
                o_cid = s.pop()
                o_ct = cell_type[o_cid]
            else:
                o_ct = 0
            i = classify_wall(ct, o_ct)
            h = ct_thick[ct][i]
            pid1, pid2 = mesh.borders(1, wid)
            l = edge_length(mesh, pos, wid)
            h_eta2 += h*(1.0/3.0)*(pos[pid1][0]**2 + pos[pid1][0]*pos[pid2][0] + pos[pid2][0]**2)*l

            if ct==14 and centroids[cid][0]>0:
                h_eta_cw += h*0.5*(pos[pid1]+pos[pid2])*l
                h_eta2_cw += h*(1.0/3.0)*(pos[pid1][0]**2 + pos[pid1][0]*pos[pid2][0] + pos[pid2][0]**2)*l
    print 'h_eta_cw, h_eta_cw/A_w', h_eta_cw, h_eta_cw / A_w
    print 'h_eta2, h_eta2/A_w', h_eta2, h_eta2 / A_w
    print 'h_eta2_cw, h_eta2/A_w', h_eta2_cw, h_eta2_cw / A_w

    plot_SVG(db, 'walls.svg')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for wid in mesh.wisps(1):
        pid1, pid2 = mesh.borders(1, wid)
        l = Line2D((pos[pid1][0], pos[pid2][0]), (pos[pid1][1], pos[pid2][1]), lw = 0.03*wall_thickness[wid])
        ax.add_line(l)
    ax.set_xlim((box_min[0], box_max[0]))
    ax.set_ylim((box_min[1], box_max[1]))
    
    plt.show()
main()

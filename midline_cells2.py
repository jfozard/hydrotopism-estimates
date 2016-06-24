
import numpy as np
from math import sin, cos, exp, sqrt, asin, atan2, pi, erfc, log
import scipy.linalg as la
from operator import itemgetter
from matplotlib.lines import Line2D
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
from random import random
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Polygon

import seaborn as sns
import pandas as pd
from math import sqrt
import itertools

sns.set_context("poster", font_scale=2.0, rc={'lines.markeredgewidth': 1})

def hydro_data():
    xl = pd.ExcelFile("141205 ht Col-0 no sorbitol.xlsx")
    df1 = xl.parse("hydrotropism", skiprows=1, index_col=False)
    df1 = df1[~df1.plate.apply(lambda x: isinstance(x, basestring))]
    df1 = df1.fillna(method='pad')

    df1.rename(columns={df1.columns[0]:"Conditions"}, inplace=True)

    df1 = df1[["Conditions", "plate",0,2,4,6,8,10,12,14]]
    df1 = df1.reset_index()

    df1['plant']=df1.reset_index().groupby(["Conditions", "plate"]).cumcount()+1
    df1 = df1.drop(u'index', axis=1)
    df1 = df1.set_index(["Conditions", "plate", "plant"])
    v = next(x for x in df1.index.get_level_values(0) if ('36' in x and 'sorbitol' in x))
    data = df1.xs(v)
    data_mean = data.mean()
    data_2se = 2*data.std()/sqrt(len(data.index))
    return data.columns, data_mean, data_2se

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, 
    in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments


# Interface to LineCollection:

def colorline(x, y, z=None, cmap=plt.get_cmap('cool'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc


def rot(v):
    """                                                                         
    rotate a 2d vector (numpy) by 90 degrees anticlockwise                    
    :param v: input vector                                                      
    :type v: Vector2/list/tuple                                                 
    :returns: Vector2 rotated v                                                 
    """
    return np.array((-v[1],v[0]))

def ml_to_xy(m, kappa, l, x0, theta0, bp_data=None):
    j = m[0]
    xi = m[1]
    eta = m[2]
    if bp_data:
        x = np.array(bp_data[0][j])
        theta = bp_data[1][j]
        t = bp_data[2][j]
    else:
        x = np.array(x0)
        theta = theta0
        for i in range(j):
            t = np.array((cos(theta), sin(theta)))
            n = np.array((-sin(theta), cos(theta)))
            if abs(kappa[i]*l[i])>1e-15:
                x += (1.0/kappa[i])*\
                    (sin(kappa[i]*l[i])*t+(2*sin(kappa[i]*l[i]/2)**2)*n)
            else:
                x += l[i]*t        
            theta += kappa[i]*l[i]             
        t = np.array((cos(theta), sin(theta)))
    n=rot(t)
    if abs(kappa[j]*l[j])>1e-15:
        x += (1/kappa[j]*sin(kappa[j]*l[j]*xi)-eta*sin(kappa[j]*l[j]*xi))*t+ \
        (1/kappa[j]*(2*sin(kappa[j]*l[j]*xi/2)**2)+eta*cos(kappa[j]*l[j]*xi))*n
    else:
#        print "t, n, xi, eta, l[j]", t, n, xi, eta, l[j]
        x += xi*l[j]*t+eta*n 
    return x


def breakpoints(kappa, l, x0, theta0):
    x = np.array(x0)
    theta = theta0
    bp = [ np.array(x) ]
    angles = [ theta ]
    tangents = [ np.array((cos(theta), sin(theta)))]
    for i in range(len(kappa)):
        t = np.array((cos(theta), sin(theta)))
        n = np.array((-sin(theta), cos(theta)))
        if abs(kappa[i]*l[i])>1e-15:
            x += (1.0/kappa[i])*(sin(kappa[i]*l[i])*t+2*sin(kappa[i]*l[i]/2)**2*n)
        else:
            x += l[i]*t        
        theta += kappa[i]*l[i]
        bp.append(np.array(x))
        angles.append(theta)
        tangents.append(np.array((cos(theta), sin(theta))))
    return bp, angles, tangents

def distance_from_apex(j, xi, l, cl=None):
    if cl is None:
        cl = np.cumsum(l)
    return cl[-1]-cl[j]+(1.0-xi)*l[j]
    
def xy_to_ml_segment(p, kappa, l, x0, t0):
    dp = p-x0
    if kappa>0:
        n0 = rot(t0)
        s = 1.0/kappa*n0-dp
        eta = (2*(dp[0]*n0[0]+dp[1]*n0[1])/kappa-(dp[0]*dp[0]+dp[1]*dp[1]))\
            / (1.0/kappa + sqrt(s[0]*s[0]+s[1]*s[1]))
        xi = asin((t0[0]*dp[0]+t0[1]*dp[1])/(1.0/kappa - eta))/kappa/l
    elif kappa<0:
        n0 = rot(t0)
        s = 1.0/kappa*n0-dp
        eta = (2*(dp[0]*n0[0]+dp[1]*n0[1])/kappa-(dp[0]*dp[0]+dp[1]*dp[1]))\
            / (1.0/kappa - sqrt(s[0]*s[0]+s[1]*s[1]))
        xi = asin((t0[0]*dp[0]+t0[1]*dp[1])/(1.0/kappa - eta))/kappa/l
    else:
        n0 = rot(t0)
        eta = n0[0]*dp[0]+n0[1]*dp[1]    #np.dot(n0, dp)
        xi = (t0[0]*dp[0]+t0[1]*dp[1])/l  #np.dot(t0, dp)/l
    return xi, eta
        
def dist(a, b) :
    return sqrt((b[1]-a[1])**2 + (b[0]-a[0])**2)
    
def nearest_point_index(p, l) :
    d=[dist(p,b) for b in l]
    minindex, minvalue = min(enumerate(d), key=itemgetter(1)) 
    return minindex
    
def xy_to_ml(p, kappa, l, x0, theta0, bp_data=None):
    if bp_data:
        bp_list, angles, tangents = bp_data
    else:
        bp_list, angles, tangents = breakpoints(kappa, l, x0, theta0)
    bp = np.asarray(bp_list)
    sides = np.array([cmp(np.dot(p-xi,ti),0.0) for xi, ti in zip(bp, tangents)])
    change_sign = np.where(sides[1:]!=sides[0:-1])
    idx = change_sign[0]
    if len(idx)>1:
        #print sides
        midpoints = 0.5*(bp[1:,:]+bp[0:-1,:]) 
        # find midpoint which is nearest to the observation points
        i = nearest_point_index(p, midpoints)
    elif len(idx)==1:
        i = idx[0]
    else:
        #print sides
        if sides[-1]==1:
            i = len(kappa)-1
        if sides[0]==-1:
            i = 0
    ml=xy_to_ml_segment(p, kappa[i], l[i], bp[i], tangents[i])
    return (i, ml[0], ml[1])

def ml_to_vel(m, kappa, l, kdot, regr, x0, theta0):
    ldot = l * regr
    v = np.array((0,0))
    
    j, xi, eta = m
    theta = theta0
    tdot = 0.0
    x = x0
    # Find velocity of breakpoint before
    for i in range(j):
        t = np.array((cos(theta), sin(theta)))
        n = np.array((-sin(theta), cos(theta)))
        if abs(kappa[i]*l[i])>1e-15:
            x1 = x + 1.0/kappa[i]*\
                (sin(kappa[i]*l[i])*t+2*sin(kappa[i]*l[i]/2)**2*n)
        else:
            x1 = x + l[i]*t        
        theta1 = theta + kappa[i]*l[i]
        t1 = np.array((cos(theta1), sin(theta1)))
        v1 = v + ldot[i]*t1
        if abs(kappa[i]*l[i])>1e-15:
            v1 += kdot[i]/kappa[i]*(l[i]*t1-(x1-x))
        else:
            v1 += 0.5*l[i]*l[i]*kdot[i]*n
        v1 += tdot*rot(x1-x)
        
        tdot += kdot[i]*l[i]+kappa[i]*ldot[i]
        v = v1
        x = x1
        theta = theta1
    t = np.array((cos(theta), sin(theta)))
    n = np.array((-sin(theta), cos(theta)))
    if abs(kappa[j]*l[j])>1e-15:
        x_p = x + (1/kappa[j]*sin(kappa[j]*l[j]*xi)-eta*sin(kappa[j]*l[j]*xi))*t\
              + (1/kappa[j]*(2*sin(kappa[j]*l[j]*xi/2)**2)+eta*cos(kappa[j]*l[j]*xi))*n
    else:
        x_p = x + xi*l[j]*t+eta*n 
    theta_p = theta + kappa[j]*l[j]*xi
    t_p = np.array((cos(theta_p), sin(theta_p)))
    v_p = v + ldot[j]*xi*t_p*(1-kappa[j]*eta) + tdot*rot(x_p-x)
    if abs(kappa[j]*l[j])>1e-15:
        v_p += kdot[j]/kappa[j]*(xi*l[j]*(1-eta*kappa[j])*t_p-1/kappa[j]*(sin(kappa[j]*l[j]*xi)*t+(2*sin(kappa[j]*l[j]*xi/2)**2)*n))
    else:
        v_p += -eta*kdot[j]*l[j]*xi*t+0.5*l[j]*l[j]*xi*xi*n*kdot[j]
    return v_p

def get_j_xi(s, l):
    j = 0
    s_tot = 0.0
    while j<len(l) and s_tot+l[j]<s:
        s_tot += l[j]
        j += 1
    if j<len(l):
        xi = (s - s_tot)/l[j]
    else:
        xi = (s - s_tot + l[-1])/l[-1]
    return j, xi


def cell_bdd(old_pt, pt, width, kappa, l, x0, theta0):
    
    bdd = []
    bdd.append(ml_to_xy((pt.n, pt.xi, width), kappa, l, x0, theta0))
    bdd.append(ml_to_xy((pt.n, pt.xi, 0), kappa, l, x0, theta0))
    bdd.append(ml_to_xy((old_pt.n, old_pt.xi, 0), kappa, l, x0, theta0))
    bdd.append(ml_to_xy((old_pt.n, old_pt.xi, width), kappa, l, x0, theta0))
    return bdd

class Point(object):
    def __init__(self, n, xi, ct=0):
        self.n = n
        self.xi = xi
        self.ct = ct
        

from lxml import etree

#Midline bending module

class Midline(object):
    def __init__(self):
        LT = 1200
        self.N = 500
        # Add properties to tissue_db?!

        self.theta0 = -pi/2
        self.x0 = np.array((0., 0.))
        self.kappa = np.array([0.0]*(self.N-1))
        self.l = np.array([LT/float(self.N) for i in range(self.N-1)])
        self.dt = 5/60.0


        self.time = 0.0

        g1 = 0.1
        g2 = 0.4
        gt = g1+(g2-g1)*0.11
        kdot = (g2-g1)*2.8/1200.0
        self.params = { 
            'growth': {'g1': g1, 'gt': gt, 'g2': g2, 'kdot': kdot,
                       's1': 400, 's2': 1000, 'delta': 50, 'T': 2 },
            'meris_length': 200.0 }
        self.width0 = 50
        
        self.lower = []
        self.upper = []

    def set_cells(self, lower, upper):
        # Note this assumes that the root is straight at this point ...
        s = np.cumsum(self.l)
        for d, l in zip(np.cumsum(lower), lower):
            n = np.searchsorted(s, s[-1]- d)
            xi = ((s[-1]-s[n-1])-d)/(s[n]-s[n-1])
            if l<15:
                ct = 0
            else:
                ct = 1
            self.lower.append(Point(n, xi, ct))

        for d, l in zip(np.cumsum(upper), upper):
            n = np.searchsorted(s, s[-1]- d)
            xi = ((s[-1]-s[n-1])-d)/(s[n]-s[n-1])
            if l<15:
                ct = 0
            else:
                ct = 1
            self.upper.append(Point(n, xi, ct))


            

    def get_smid(self):
        s = np.cumsum(self.l)
        s_mid = 0.5*s
        s_mid[1:] += 0.5*s[:-1]
        s_mid = s[-1]-s_mid
        return s_mid

    # Find midpoint of each section
    def get_regr_kdot(self):
        s_mid = np.array(self.get_smid())
        p = self.params['growth']
        if self.time < p['T']:
            g = np.zeros(s_mid.shape)
            g[s_mid<p['s2']] = p['g2']
            g[s_mid<p['s1']] = p['gt']
            g[s_mid<(p['s1']-p['delta'])] = p['g1']
            kdot = np.zeros(s_mid.shape)
            kdot[np.logical_and((p['s1']-p['delta'])<s_mid,
                                s_mid<p['s1'])] = p['kdot']
        else:
            g = np.zeros(s_mid.shape)
            g[s_mid<p['s2']] = p['g2']
            g[s_mid<p['s1']] = p['g1']
            kdot = np.zeros(s_mid.shape)            
        return g, kdot

    def step(self):
        self.refine()
        Gamma, kdot = self.get_regr_kdot()
        dt = self.dt
        self.l = self.l*np.exp(dt*Gamma)
        self.kappa += dt*kdot
        self.time += self.dt
        
    def width(self, x):
        L = self.params['meris_length']
        if x>=L:
            return self.width0
        else:
            return self.width0*sqrt(1.0-(x-L)*(x-L)/(L*L))

    def split_segment(self, i):
        kappa = self.kappa[i]
        l = self.l[i]
        self.kappa=np.insert(self.kappa, i, kappa)
        self.l[i]*=0.5
        self.l=np.insert(self.l, i, 0.5*l)

        
        for c in itertools.chain(self.lower, self.upper):
            if c.n == i:
                if c.xi<0.5:
                    c.xi *=2
                else:
                    c.n +=1
                    c.xi *=0.5

        #self.markers.insert(i+1, 0)
        

    def refine(self, l_max = 20):
        refine = np.where(self.l>l_max)
        if refine[0].size>0:
             for i in reversed(refine[0]):
                 self.split_segment(i)

    def plot(self):
        Ns = 1
        h = 0.1
        m = []
        for i in range(len(self.l)):
            for s in range(Ns):
                m.append(ml_to_xy((i, float(s)/Ns, 0), self.kappa, self.l, self.x0, self.theta0))
        m.append(ml_to_xy((len(self.l)-1, 1., 0), self.kappa, self.l, self.x0, self.theta0))

        plt.clf()
        plt.hold(True)
        x=[p[0] for p in m]
        y=[p[1] for p in m]
        
        bp, angles, tangents= breakpoints(self.kappa, self.l, self.x0, self.theta0)
        x=[p[0] for p in bp]
        y=[p[1] for p in bp]
        plt.plot(x[::10],y[::10],'k-')
#        xm = [p[0] for i, p in enumerate(bp) if self.markers[i]==1]
#        ym = [p[1] for i, p in enumerate(bp) if self.markers[i]==1]


#        plt.plot(xm,ym,'kx')

        plt.xlim((-100, 3000))
        plt.ylim((-3000, 100))
        plt.draw()




    def length(self):
        return sum(self.l)

    def growth_rate(self):
        regr, kdot = self.get_regr_kdot()
        return sum(l*g for l,g in zip(self.l, regr))

    def tip_angle(self):
        return self.theta0+sum(l*k for l,k in zip(self.l, self.kappa))

def conv_number(x):
    try:
        return float(x)
    except ValueError:
        return None

#plt.ion()
def main():

    m = Midline()

    tip_angles = [m.tip_angle()+pi/2]
    regr_history, kdot_history, s_history, kappa_history = [], [], [], []
    hydro_history = []
    length_history = []
    Ni = int(2.0/m.dt)

#    plt.figure()
    for i in range(int(12.0/m.dt)):
        s = m.get_smid()
        kappa = np.array(m.kappa)
        regr, kdot = m.get_regr_kdot()
        m.step()        
#        m.plot()
        tip_angles.append(m.tip_angle()+pi/2)
        regr_history.append(regr)
        kdot_history.append(kdot)
        kappa_history.append(kappa)
        s_history.append(s)
        length_history.append(m.length())
#    plt.xlim([0,4000])
#    plt.ylim([500, 4500])
#    plt.gca().autoscale_view()
#    plt.ioff()
    labels, data_mean, data_2se = hydro_data()    

    plt.figure()
    plt.plot(m.dt*np.arange(len(tip_angles)),np.array(tip_angles)*180.0/pi)
    plt.errorbar(labels[:-1], data_mean[:-1], yerr=data_2se[:-1])
    plt.xlim(0,12)
    ax = plt.gca()
    ax.set_xticks([0, 12])
    ax.set_xticks(np.arange(0,12,2), minor=True)
    ax.set_yticks([0, 40])
    ax.set_yticks(np.arange(0, 60, 10), minor=True)
    plt.xlabel("time (hr)")
    plt.ylabel("tip angle (degrees)")

    plt.savefig('midline_cells.svg')

if __name__=="__main__":
    main()
    

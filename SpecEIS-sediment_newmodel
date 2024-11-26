import os    
os.environ["OMP_NUM_THREADS"] = '1' 
import matplotlib
import firedrake as df
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import ufl

# HELPER FUNCTIONS
def Max(a, b): return (a+b+abs(a-b))/df.Constant(2)

def Min(a, b): return (a+b-abs(a-b))/df.Constant(2)

def softplus(y1,y2,alpha=1):
    # The softplus function is a differentiable approximation
    # to the ramp function.  Its derivative is the logistic function.
    # Larger alpha makes a sharper transition.
    return Max(y1,y2) + (1./alpha)*df.ln(1+df.exp(alpha*(Min(y1,y2)-Max(y1,y2))))

from numpy.polynomial.legendre import leggauss
def full_quad(order):
    # This function provides the points and weights for
    # Gaussian quadrature (here used in the vertical dimension)
    points,weights = leggauss(order)
    points = (points+1)/2.
    weights /= 2.
    return points,weights

# Logistic function
sigmoid = lambda z: 1./(1+df.exp(-z))

L = 45000.                  # Characteristic domain length
#L = 90000
spy = 60**2*24*365          
thklim = 1.0                # Minimum Ice Thickness

zmin = -300.0               # Minimum elevation
zmax = 2200.0               # Maximum elevation

amin = df.Constant(-7.0)       # Minimum smb
amax = df.Constant(5.0)        # Maximum smb

c = 2.0                     # Coefficient of exponential decay

amp = 50.0                   # Amplitude of sinusoidal topography 

rho = rho_i = 917.                  # Ice density
rho_w = 1029.0              # Seawater density
rho_s = 1600.0              # Sediment density
rho_r = 2750.0              # Bedrock density

La = 3.35e5

g = 9.81                    # Gravitational acceleration
n = 3.0                     # Glen's exponent
m = 1.0                     # Sliding law exponent
b = 1e-16**(-1./n)          # Ice hardness
eps_reg = df.Constant(1e-4)    # Regularization parameter

l_s = df.Constant(2.0)         # Sediment thickness at which bedrock erosion becomes negligible 
be = df.Constant(2e-9)         # Bedrock erosion coefficient
cc = df.Constant(1e-11)        # Fluvial erosion coefficient
d = df.Constant(500.0)         # Fallout fraction
h_0 = df.Constant(0.1)         # Subglacial cavity depth  


k = df.Constant(0.7)           # Fraction of overburden defining water pressure

# Shelf collapse factor - units of a^{-1}
# Fraction of floating ice removed per year
scf = df.Constant(0.2)

k_diff = df.Constant(20)       # Sediment diffusivity
h_ref = df.Constant(10)        # Thickness at which diffusivity stops increasing: sediment diffusive fluxes need to go
                               # to zero as sediment thickness goes to zero, which implies that the flux should be 
                               # proportional to the sed thickness - however, we also don't want thicker moraines to
                               # diffuse faster than thin ones, because we don't think that's how hillslope processes
                               # work.  As such the sediment flux is proportional to 1 - e^-(h/h_ref), which approximates
                               # a thickness-linear flux for thin sediment and thickness-independent flux for thick sediment.  


dt_float = 0.1           # Time step
dt = df.Constant(dt_float)


##########################################################
################           MESH          #################
##########################################################  

# Define a rectangular mesh
nx = 300
mesh = df.IntervalMesh(nx,0,2*L)
Q_dg = df.FunctionSpace(mesh,"DG",0)
H = df.Function(Q_dg)

x, = df.SpatialCoordinate(mesh)

surface_expression = (zmax - zmin)*df.exp(-3*x/L)  + zmin - amp*df.sin(4*np.pi*x/L) + thklim
bed_expression = (zmax - zmin)*df.exp(-3*x/L)  + zmin - amp*df.sin(4*np.pi*x/L)
beta_expression = df.Constant(50.0)


#########################################################
#################  FUNCTION SPACES  #####################
#########################################################

nhat = df.FacetNormal(mesh)[0]

# CG1 Function Space
E_cg = df.FiniteElement("CG",mesh.ufl_cell(),1)
Q_cg = df.FunctionSpace(mesh,E_cg)

# DG0 Function Space
E_dg = df.FiniteElement("DG",mesh.ufl_cell(),0)
Q_dg = df.FunctionSpace(mesh,E_dg)

# Mixed element for coupled velocity-thickness solve
# (depth-averaged velocity, deformational velocity, DG0 thickness, CG1 thickness projection)
E_glac = df.MixedElement([E_cg,E_cg,E_dg])  
V_g = df.FunctionSpace(mesh,E_glac)

# Mixed element for coupled sediment stuff
# (Bedrock elevation, fluvial sediment flux, sediment thickness, effective subglacial cavity height
E_sed = df.MixedElement([E_dg,E_dg,E_dg,E_dg])
V_sed = df.FunctionSpace(mesh,E_sed)


# Mixed element for sediment diffusion (handled separately)
# (sediment thickness, sediment velocity)
E_dif = df.MixedElement([E_dg,E_dg])
V_dif = df.FunctionSpace(mesh,E_dif)

#########################################################
#################  FUNCTIONS  ###########################
#########################################################
np.random.seed(0)

# Velocity and thickness functions
U = df.Function(V_g)
dU = df.TrialFunction(V_g)
Phi = df.TestFunction(V_g)

# Split into components 
# depth-averaged velocity, deformation velocity, thickness
ubar,udef,H = df.split(U)
phibar,phidef,xsi = df.split(Phi)

ubar0 = df.Function(Q_cg)
udef0 = df.Function(Q_cg)
H0 = df.Function(Q_dg)

ubar0.dat.data[:] += 1e-10
udef0.dat.data[:] += 1e-10
H0.vector()[:] = 25

U.sub(0).assign(ubar0)
U.sub(1).assign(udef0)
U.sub(2).assign(H0)

# Sediment functions
T = df.Function(V_sed)
dT = df.TrialFunction(V_sed)
Psi = df.TestFunction(V_sed)

# Split into components
# Bed elevation, fluvial sed flux, sed thickness, water layer thickness
B,Qs,h_s,h_eff = df.split(T)
psi_B,psi_Q,psi_h,psi_eff = df.split(Psi)

B0 = df.Function(Q_dg)
Qs0 = df.Function(Q_dg)
h_s0 = df.Function(Q_dg)
h_eff0 = df.Function(Q_dg)

B0.interpolate(bed_expression)
h_s0.dat.data[:] = 1e-1
h_eff0.vector()[:] = 1.0

T.sub(0).assign(B0)
T.sub(2).assign(h_s0)
T.sub(3).assign(h_eff0)

# Sediment diffusion functions
P = df.Function(V_dif)
dP = df.TrialFunction(V_dif)
Tau = df.TestFunction(V_dif)

# Split into components
# sed thickness, sed velocity
h_sd,q_d = df.split(P)
w_d,tau_d = df.split(Tau)

h_sd0 = df.Function(Q_dg)

# Scalar test functions for uncoupled water flux
psi = df.TestFunction(Q_dg)
dQ = df.TrialFunction(Q_dg)

# Derived geometric quantities
Bhat = B + h_s                               # Top of sediment            

l = softplus(df.Constant(0),Bhat,alpha=5.0)  # Water surface, or the greater of
                                             # bedrock topography or zero

Base = softplus(Bhat,-rho/rho_w*H,alpha=5.0) # Ice base is the greater of the 
                                             # bedrock topography or the base of 
                                             # the shelf

D = softplus(-Bhat,df.Constant(0),alpha=5.0) # Water depth
S = Base + H                                 # Ice surface

# Basal traction
beta2 = df.interpolate(beta_expression,Q_cg)

# Surface mass balance
climate_factor = df.Constant(1)      
adot = climate_factor*(8 - 16/(4*L/3.)*x)

########################################################
#################   MOMENTUM BALANCE   #################
########################################################

class VerticalBasis(object):
    def __init__(self,u,coef,dcoef):
        self.u = u
        self.coef = coef
        self.dcoef = dcoef

    def __call__(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.coef)])

    def ds(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.dcoef)])

    def dx(self,s,x):
        return sum([u.dx(x)*c(s) for u,c in zip(self.u,self.coef)])

class VerticalIntegrator(object):
    def __init__(self,points,weights):
        self.points = points
        self.weights = weights
    def integral_term(self,f,s,w):
        return w*f(s)
    def intz(self,f):
        return sum([self.integral_term(f,s,w) for s,w in zip(self.points,self.weights)])

def dsdx(s):
    return 0#1./H_*(S.dx(0) - s*H_.dx(0))

def dsdz(s):
    return -1./H

# ANSATZ    
p = 4.0
coef = [lambda s:1.0, lambda s:1./p*((p+1)*s**p-1.)]
dcoef = [lambda s:0.0, lambda s:(p+1)*s**(p-1)]

u_ = [ubar,udef]
u0_ = [ubar0,udef0]
phi_ = [phibar,phidef]

u = VerticalBasis(u_,coef,dcoef)
phi = VerticalBasis(phi_,coef,dcoef)

def eta_v(s):
    return b/2.*((u.dx(s,0) + u.ds(s)*dsdx(s))**2 \
                +0.25*((u.ds(s)*dsdz(s))**2) \
                + eps_reg)**((1.-n)/(2*n))

def membrane_xx(s):
    return (phi.dx(s,0) + phi.ds(s)*dsdx(s))*H*eta_v(s)*(4*(u.dx(s,0) + u.ds(s)*dsdx(s)))

def shear_xz(s):
    return dsdz(s)**2*phi.ds(s)*H*eta_v(s)*u.ds(s)

points,weights = full_quad(4) 
vi = VerticalIntegrator(points,weights)

# Pressure and sliding law
P_0 = H                                               # Overburden pressure
P_w = ufl.max_value(k*H,rho_w/rho_i*(l-Base))         # Water pressure
N = ufl.max_value(P_0-P_w,df.Constant(0.000))         # Effective pressure

# Nonstandard driving stress term
driving_stress = -df.Dx(rho*g*H*phibar,0)*S*df.dx + df.jump(rho*g*H*phibar,nhat)*df.avg(S)*df.dS + rho*g*H*phibar*nhat*S*df.ds

# Momentum balance weak form
I_stress = (- vi.intz(membrane_xx) - vi.intz(shear_xz) - phi(1)*beta2*u(1)*N)*df.dx - driving_stress

#############################################################################
##########################  MASS BALANCE  ###################################
#############################################################################

# Jumps and averages for DG0 method
H_avg = 0.5*(H('+') + H('-'))
H_jump = H('+')*nhat('+') + H('-')*nhat('-')
xsi_avg = 0.5*(xsi('+') + xsi('-'))
xsi_jump = xsi('+')*nhat('+') + xsi('-')*nhat('-')

uvec = df.as_vector([ubar,])
unorm = (df.dot(uvec,uvec))**0.5
uH = df.avg(ubar)*H_avg + 0.5*df.avg(unorm)*H_jump

# Flotation indicator
floating = df.conditional(
                df.lt(H, rho_w/rho_i*(0 - Bhat)),
                df.Constant(1.0),df.Constant(0.0))


I_transport = ((H-H0)/dt - adot + scf*H*floating)*xsi*df.dx + df.dot(uH,xsi_jump)*df.dS + ubar*H*nhat*xsi*df.ds(2)

# Weak form of coupled velocity/thickness solve
R = I_stress + I_transport

#############################################################################
###########################  Water Flux  ####################################
#############################################################################

# Meltrate
me = (beta2*N*u(1)**2/(rho*La) - Min(adot,-1e-16))*sigmoid(H-(thklim+df.Constant(1)))

#h = df.CellDiameter(mesh)

dQ_avg = 0.5*(dQ('+') + dQ('-'))
dQ_jump = dQ('+')*nhat('+') + dQ('-')*nhat('-')
psi_avg = 0.5*(psi('+') + psi('-'))
psi_jump = psi('+')*nhat('+') + psi('-')*nhat('-')

dQ_upwind = dQ_avg + 0.5*dQ_jump

Qw = df.Function(Q_dg)
R_Qw = -me*psi*df.dx + df.dot(dQ_upwind,psi_jump)*df.dS + dQ*nhat*psi*df.ds(2)
A_Qw = df.lhs(R_Qw)
b_Qw = df.rhs(R_Qw)


#############################################################################
#############################  Sediment evolution  ##########################
#############################################################################

delta = df.exp(-h_s/l_s) # mantling effect where presence of sediment thk will yield no bed erosion 
ubar_w = Qw/h_eff        # Water velocity

Bdot = -be*beta2*N*u(1)**2*delta     # Rate of bedrock erosion
edot = cc/h_eff*ubar_w**2*(1-delta)  # Erosion rate
ddot = d*Qs/Qw                       # Deposition rate

Qs_avg = 0.5*(Qs('+') + Qs('-'))
Qs_jump = Qs('+')*nhat('+') + Qs('-')*nhat('-')
psiQ_avg = 0.5*(psi_Q('+') + psi_Q('-'))
psiQ_jump = psi_Q('+')*nhat('+') + psi_Q('-')*nhat('-')

Qs_upwind = Qs_avg + 0.5*Qs_jump

h = df.CellDiameter(mesh)

R_Qs = (ddot - edot)*psi_Q*df.dx + df.dot(Qs_upwind,psiQ_jump)*df.dS
R_hs = psi_h*((h_s - h_s0)/dt + rho_r/rho_s*Bdot - ddot + edot)*df.dx
R_B = psi_B*((B-B0)/dt - Bdot)*df.dx
R_heff = psi_eff*(h_eff - softplus(h_0,Base-Bhat,alpha=5.0))*df.dx

# Weak form of sediment dynamics, solves for bedrock elevation, fluvial sed. flux,
# sediment thickness, projected sediment thickness, and effective water layer thickness.
R_sed = R_B + R_Qs + R_hs + R_heff
J_sed = df.derivative(R_sed,T,dT)


#############################################################################
###########################  Sediment diffusion  ############################
#############################################################################

f = df.Constant(1) - df.exp(-h_sd/h_ref)
hd_avg = 0.5*(h_sd('+') + h_sd('-'))
hd_jump = h_sd('+')*nhat('+') + h_sd('-')*nhat('-')

qvec = df.as_vector([q_d,])
qnorm = (df.dot(qvec,qvec) + 1e-10)**0.5
qH = df.avg(q_d)*hd_avg + 0.5*df.avg(qnorm)*hd_jump

R_dh = w_d*(h_sd - h_sd0)/dt*df.dx + df.jump(w_d,nhat)*qH*df.dS# + w_d*nhat*q_d*df.ds
R_dq = tau_d*q_d*df.dx - k_diff*df.Dx(tau_d*f,0)*(h_sd + B)*df.dx + df.jump(tau_d*f,nhat)*k_diff*df.avg(h_sd + B)*df.dS + tau_d*nhat*k_diff*f*(h_sd + B)*df.ds 

R_d = R_dh + R_dq

##############################################################################
##########################  Solvers  #########################################
##############################################################################

solver_parameters_mass ={"snes_type": 'vinewtonrsls',
                        "pc_factor_mat_solver_type": "mumps",
                        "snes_rtol": 1.e-4,
                        "snes_atol": 1.e-3,  
                     	"snes_max_it": 10,
                    	"report": True,
                        "snes_monitor": None,
                     	"error_on_nonconvergence": True}

mass_problem = df.NonlinearVariationalProblem(R, U)

lower_bounds = df.Function(V_g)
upper_bounds = df.Function(V_g)
lower_bounds.sub(0).assign(df.Constant(-1e10))
lower_bounds.sub(1).assign(df.Constant(-1e10))
lower_bounds.sub(2).assign(thklim)

upper_bounds.sub(0).assign(df.Constant(1e10))
upper_bounds.sub(1).assign(df.Constant(1e10))
upper_bounds.sub(2).assign(1e10)

mass_solver = df.NonlinearVariationalSolver(mass_problem, solver_parameters=solver_parameters_mass)

solver_parameters_sed ={"snes_type": 'vinewtonrsls',
                        "pc_factor_mat_solver_type": "mumps",
                        "snes_rtol": 1.e-4,
                        "snes_atol": 1.e-3,  
                     	"snes_max_it": 10,
                    	"report": True,
                        "snes_monitor": None,
                     	"error_on_nonconvergence": True}

sed_problem = df.NonlinearVariationalProblem(R_sed, T)

lower_bounds_sed = df.Function(V_sed)
upper_bounds_sed = df.Function(V_sed)
lower_bounds_sed.sub(0).assign(df.Constant(-1e10))
lower_bounds_sed.sub(1).assign(df.Constant(-1e10))
lower_bounds_sed.sub(2).assign(-1e10)
lower_bounds_sed.sub(3).assign(-1e10)

upper_bounds_sed.sub(0).assign(df.Constant(1e10))
upper_bounds_sed.sub(1).assign(df.Constant(1e10))
upper_bounds_sed.sub(2).assign(1e10)
upper_bounds_sed.sub(3).assign(1e10)

sed_solver = df.NonlinearVariationalSolver(sed_problem, solver_parameters=solver_parameters_sed)

###############################################################################
#########################  Animation Init.  ###################################
###############################################################################

plt.ion()
fig, ax = plt.subplots()

xx = df.interpolate(x,Q_dg).dat.data[:]
ph_bed, = ax.plot(xx,B0.dat.data[:],'k-')
base = df.interpolate(Base,Q_dg)
ph_base, = ax.plot(xx,base.dat.data[:],'b-')
surf = df.interpolate(S,Q_dg)
ph_surf, = ax.plot(xx,surf.dat.data[:],'b-')
sed = df.interpolate(B0 + h_s0,Q_dg)
ph_sed, = ax.plot(xx,sed.dat.data[:],'g-')


##############################################################################
############################  Run Model  #####################################
##############################################################################

# Time interval
t = 0.0
t_end = 30000

counter = 0

# Maximum time step!!  Increase with caution.
dt_max = 5.0

# Loop over time
while t<t_end:
    try: # If the solvers don't converge, reduce the time step and try again.
        print(t,dt_float)
        
        if counter%1==0:
            fig.canvas.start_event_loop(0.00001)
            fig.canvas.draw_idle()

        # Solve the velocity-ice thickness equations
        mass_solver.solve(bounds=(lower_bounds,upper_bounds))

        # Solve for water flux
        df.solve(A_Qw == b_Qw,Qw)

        # Solve the sediment equations (excluding diffusion)
        sed_solver.solve(bounds=(lower_bounds_sed,upper_bounds_sed))

        # Update the previous thickness
        ubar0.dat.data[:] = U.dat.data[0]
        udef0.dat.data[:] = U.dat.data[1]
        H0.dat.data[:] = U.dat.data[2]
        
        # Update previous bed elevation
        B0.dat.data[:] = T.dat.data[0]

        # Update previous sediment thickness
        h_s0.dat.data[:] = T.dat.data[2]

        # diffusion mechanics
        # Set current sed thickness to solution from sed transport
        h_sd0.dat.data[:] = h_s0.dat.data[:]
        P.sub(0).assign(h_s0)

        # Solve diffusion equation
        df.solve(R_d==0,P)

        # Update main sed thickness variables
        h_s0.dat.data[:] = P.dat.data[0]
        T.sub(2).assign(h_s0)

        # Increase time step if solvers complete successfully
        dt_float = min(1.05*dt_float,dt_max)
        dt.assign(dt_float)

        # perform plotting
        base = df.interpolate(Base,Q_dg)
        surf = df.interpolate(S,Q_dg)
        sed = df.interpolate(B0 + h_s0,Q_dg)

        ph_bed.set_ydata(B0.dat.data[:])
        ph_base.set_ydata(base.dat.data[:])
        ph_surf.set_ydata(surf.dat.data[:])
        ph_sed.set_ydata(sed.dat.data[:])

        t+=dt_float
        counter+=1

    except df.ConvergenceError:

        # If the solvers don't converge, reset variables to previous time step's succesful values and try again.  
        U.sub(0).assign(1e-10)
        U.sub(1).assign(1e-10)
        U.sub(2).assign(H0)

        T.sub(0).assign(B0)
        T.sub(1).assign(df.Constant(0.0))
        T.sub(2).assign(h_s0)
        T.sub(3).assign(h_eff0)

        dt_float/=2.
        dt.assign(dt_float)
        print('convergence failed, reducing time step and trying again')

    if (dt_float < 1e-10): # Things have really gone off the rails and the simulation is shutting down
        print("model stalling, time step approaching zero")
        break

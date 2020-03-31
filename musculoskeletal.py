import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sklearn.linear_model import Ridge
from scipy.special import expit
from scipy.integrate import solve_ivp
from regression import Regression
from activation import Activation
from regression import load_data, get_norm_emg

### Hill Type Muscle Model

class HillTypeMuscle:
    """
    Damped Hill-type muscle model adapted from Millard et al. (2013). The
    dynamic model is defined in terms of normalized length and velocity.
    To model a particular muscle, scale factors are needed for force, CE
    length, and SE length. These are given as constructor arguments.
    """

    def __init__(self, f0M, resting_length_muscle, resting_length_tendon):
        """
        :param f0M: maximum isometric force
        :param resting_length_muscle: actual length (m) of muscle (CE) that corresponds to
            normalized length of 1
        :param resting_length_tendon: actual length of tendon (m) that corresponds to
            normalized length of 1
        """
        self.f0M = f0M
        self.resting_length_muscle = resting_length_muscle
        self.resting_length_tendon = resting_length_tendon

    def norm_tendon_length(self, muscle_tendon_length, normalized_muscle_length):
        """
        :param muscle_tendon_length: non-normalized length of the full muscle-tendon
            complex (typically found from joint angles and musculoskeletal geometry)
        :param normalized_muscle_length: normalized length of the contractile element
            (the state variable of the muscle model)
        :return: normalized length of the tendon
        """
        return (muscle_tendon_length - self.resting_length_muscle * normalized_muscle_length) / self.resting_length_tendon

    def get_force(self, total_length, norm_muscle_length):
        """
        :param total_length: muscle-tendon length (m)
        :param norm_muscle_length: normalized length of muscle (the state variable)
        :return: muscle tension (N)
        """
        return self.f0M * force_length_tendon(self.norm_tendon_length(total_length, norm_muscle_length))
    

def velocity_eq(v, *data):
    """
    :param v: normalized lengthening velocity of muscle (contractile element)
    :param data: contains relevant parameters for velocity calculetion
        a: activation 
        lm: normalized length of muscle (contractile element)
        lt: normalized length of tendon (series elastic element)
        beta: dampening coefficient
    :return: damped musculotendon equilibrium equation 
    """
    
    alpha = 0 # pennation angle
    a,lm,lt,beta = data # unpack parameters

    fMuscle = force_length_muscle(lm) # find force from CE force-length regression curve
    fPE = force_length_parallel(lm) # calculate force from PE force-length relationship
    fTendon = force_length_tendon(lt) # calculate force from SE force-length relationship

    # Return Equation 8 from Millard et. al.
    return (a * fMuscle * force_velocity_muscle(v) + fPE + beta*v) * np.cos(alpha) - fTendon


def get_velocity(a, lm, lt):
    """
    :param a: activation (between 0 and 1)
    :param lm: normalized length of muscle (contractile element)
    :param lt: normalized length of tendon (series elastic element)
    :return: normalized lengthening velocity of muscle (contractile element)
    """

    beta = 0.1 # damping coefficient

    vel = fsolve(velocity_eq, 0.0, args=(a,lm,lt,beta)) # numerically solve for velocity
    return vel 


def force_length_tendon(lt):
    """
    :param lt: normalized length of tendon (series elastic element)
    :return: normalized tension produced by tendon
    """
    lt_s = 1.0 # slack length of series element

    if isinstance(lt,np.ndarray): # return an ndarray if input is an ndarray
        norm = np.zeros(len(lt))
        for i in range(len(lt)):
            if lt[i] < lt_s:
                norm[i] = 0.0
            else: 
                norm[i] = 10.0 * (lt[i] - lt_s) + 240.0 * (lt[i] - lt_s)**2
        return norm
    else: # If the input is not an ndarray, then we just return the scalar value 
          # Just checking for floats is good enough for this assignmennt
        if lt < lt_s:
            return 0.0
        else: 
            return 10.0 * (lt - lt_s) + 240.0 * (lt - lt_s)**2


def force_length_parallel(lm):
    """
    :param lm: normalized length of muscle (contractile element)
    :return: normalized force produced by parallel elastic element
    """
    lpe_s = 1.0 # slack length of series element
    
    if isinstance(lm,np.ndarray): # return an ndarray if input is an ndarray
        norm = np.zeros(len(lm))
        for i in range(len(lm)):
            if lm[i] < lpe_s:
                norm[i] = 0.0
            else: 
                norm[i] = (3.0 * (lm[i] - lpe_s)**2) / (0.6 + lm[i] - lpe_s)
        return norm
    else: # If the input is not an ndarray, then we just return the scalar value 
          # Just checking for floats is good enough for this assignmennt
        if lm < lpe_s:
            return 0.0
        else: 
            return (3.0 * (lm - lpe_s)**2) / (0.6 + lm - lpe_s)


def get_muscle_force_velocity_regression():
    data = np.array([
        [-1.0028395556708567, 0.0024834319945283845],
        [-0.8858611825192801, 0.03218792009622429],
        [-0.5176245843258415, 0.15771090304473967],
        [-0.5232565269687035, 0.16930496922242444],
        [-0.29749770052593094, 0.2899790099290114],
        [-0.2828848376217543, 0.3545364496120378],
        [-0.1801231103040022, 0.3892195938775034],
        [-0.08494610976156225, 0.5927831890757294],
        [-0.10185137142991896, 0.6259097662790973],
        [-0.0326643239546236, 0.7682365981934388],
        [-0.020787245583830716, 0.8526638522676352],
        [0.0028442725407418212, 0.9999952831301149],
        [0.014617579774061973, 1.0662107025777694],
        [0.04058866536166583, 1.124136223202283],
        [0.026390887007381902, 1.132426122025424],
        [0.021070257776939272, 1.1986556920827338],
        [0.05844673474682183, 1.2582274002971627],
        [0.09900238201929201, 1.3757434966156459],
        [0.1020023112662436, 1.4022310794556732],
        [0.10055894908138963, 1.1489210160137733],
        [0.1946227683309354, 1.1571212943090965],
        [0.3313459588217258, 1.152041225442796],
        [0.5510200231126625, 1.204839508502158]
    ])

    velocity = data[:,0]
    force = data[:,1]

    centres = np.arange(-1, 0, .2)
    width = .15
    result = Regression(velocity, force, centres, width, .1, sigmoids=True)

    return result


def get_muscle_force_length_regression():
    """
    CE force-length data samples from Winters et al. (2011) Figure 3C,
    normalized so that max force is ~1 and length at max force is ~1.
    The sampples were taken form the paper with WebPlotDigitizer, and
    cut-and-pasted here.

    1) Use WebPlotDigitizer to extract force-length points
    from Winters et al. (2011) Figure 3C, which is on Learn. Click
    "View Data", select all, cut, and paste below. 2) Normalize the data
    so optimal length = 1 and peak = 1. 3) Return a Regression object that
    uses Gaussian basis functions. 
    """
    data = np.array([
            [38.39580209895055, 14.611872146118998],
            [39.35532233883059, 3.6529680365296713],
            [37.37631184407801, 9.817351598173673],
            [41.754122938530735, 1.8264840182648072],
            [40.37481259370315, 17.80821917808217],
            [41.39430284857571, 14.611872146118714],
            [41.39430284857571, 15.75342465753424],
            [39.23538230884557, 24.42922374429223],
            [40.37481259370314, 21.689497716894962],
            [41.39430284857571, 26.712328767123296],
            [42.89355322338831, 23.74429223744292],
            [43.373313343328334, 23.74429223744292],
            [43.85307346326836, 22.146118721461193],
            [41.39430284857571, 31.7351598173516],
            [42.05397301349325, 31.96347031963471],
            [40.37481259370314, 36.757990867579906],
            [43.373313343328334, 34.93150684931507],
            [42.53373313343328, 42.00913242009126],
            [43.373313343328334, 44.74885844748857],
            [44.3928035982009, 45.43378995433791],
            [45.53223388305847, 46.34703196347034],
            [45.532233883058474, 43.60730593607306],
            [46.671664167916035, 44.74885844748857],
            [42.77361319340329, 46.347031963470315],
            [42.8335832083958, 48.401826484018244],
            [43.07346326836581, 50.228310502283094],
            [43.373313343328334, 53.88127853881278],
            [43.673163418290855, 57.07762557077628],
            [45.172413793103445, 53.88127853881278],
            [45.35232383808096, 53.65296803652967],
            [44.3928035982009, 60.502283105022826],
            [46.371814092953514, 62.557077625570756],
            [47.39130434782609, 62.557077625570756],
            [48.95052473763119, 63.013698630136986],
            [47.691154422788614, 66.66666666666666],
            [45.71214392803598, 67.57990867579908],
            [45.89205397301349, 70.77625570776253],
            [46.37181409295353, 71.46118721461184],
            [47.39130434782608, 71.46118721461187],
            [46.37181409295353, 73.51598173515978],
            [46.67166416791604, 75.34246575342465],
            [47.03148425787106, 80.59360730593608],
            [47.45127436281858, 81.50684931506848],
            [47.5712143928036, 81.50684931506848],
            [48.11094452773613, 83.1050228310502],
            [48.95052473763119, 81.05022831050226],
            [48.95052473763119, 81.9634703196347],
            [49.430284857571216, 76.02739726027397],
            [50.56971514242879, 74.42922374429222],
            [50.749625187406295, 77.85388127853881],
            [50.92953523238381, 79.90867579908675],
            [51.10944527736132, 78.5388127853881],
            [49.73013493253374, 81.73515981735159],
            [48.89055472263868, 85.15981735159815],
            [49.79010494752623, 84.93150684931504],
            [50.629685157421285, 84.93150684931504],
            [49.61019490254873, 86.52968036529678],
            [50.26986506746627, 87.4429223744292],
            [50.629685157421285, 86.98630136986299],
            [50.629685157421285, 90.41095890410958],
            [51.46926536731634, 89.49771689497715],
            [51.64917541229386, 90.86757990867578],
            [52.608695652173914, 89.04109589041093],
            [53.26836581709145, 88.58447488584473],
            [53.50824587706146, 83.33333333333331],
            [53.50824587706147, 78.76712328767121],
            [53.86806596701649, 92.0091324200913],
            [53.62818590704648, 92.23744292237441],
            [53.44827586206897, 94.52054794520546],
            [53.50824587706147, 96.3470319634703],
            [53.74812593703148, 96.57534246575341],
            [54.287856071964015, 94.52054794520546],
            [54.10794602698651, 93.83561643835614],
            [54.64767616191904, 99.31506849315066],
            [55.72713643178411, 96.11872146118719],
            [56.086956521739125, 96.11872146118719],
            [56.38680659670165, 99.54337899543377],
            [56.74662668665667, 99.54337899543377],
            [56.98650674662669, 99.08675799086757],
            [57.22638680659669, 97.71689497716892],
            [57.46626686656672, 99.54337899543377],
            [57.766116941529226, 99.31506849315066],
            [58.545727136431786, 96.11872146118719],
            [58.90554722638681, 99.08675799086757],
            [59.32533733133433, 97.71689497716892],
            [59.56521739130435, 95.66210045662098],
            [59.74512743628186, 96.57534246575341],
            [59.925037481259366, 95.43378995433787],
            [60.52473763118441, 99.54337899543377],
            [60.52473763118441, 93.37899543378992],
            [57.76611694152924, 91.09589041095887],
            [58.42578710644677, 90.63926940639267],
            [59.32533733133433, 91.09589041095887],
            [61.364317841079455, 87.67123287671231],
            [61.30434782608695, 91.7808219178082],
            [61.30434782608695, 94.74885844748857],
            [62.32383808095952, 96.3470319634703],
            [62.143928035982015, 89.26940639269404],
            [61.24437781109445, 84.24657534246573],
            [61.36431784107946, 79.45205479452054],
            [61.36431784107946, 76.71232876712327],
            [62.56371814092954, 79.68036529680364],
            [63.34332833583209, 80.36529680365297],
            [63.403298350824585, 79.45205479452054],
            [63.82308845577211, 80.13698630136989],
            [64.48275862068965, 81.73515981735159],
            [63.58320839580209, 85.38812785388126],
            [63.103448275862064, 86.07305936073058],
            [63.6431784107946, 89.49771689497715],
            [63.7631184407796, 86.75799086757988],
            [64.48275862068965, 86.75799086757988],
            [65.08245877061469, 76.25570776255705],
            [63.943028485757125, 76.02739726027399],
            [66.34182908545728, 75.57077625570778],
            [65.5622188905547, 72.83105022831052],
            [66.10194902548726, 72.14611872146119],
            [65.68215892053972, 68.2648401826484],
            [65.74212893553224, 66.21004566210044],
            [65.32233883058471, 63.92694063926942],
            [66.8215892053973, 66.21004566210046],
            [67.18140929535232, 63.24200913242012],
            [67.72113943028486, 62.55707762557081],
            [63.343328335832084, 52.96803652968036],
            [64.72263868065967, 53.42465753424658],
            [64.72263868065967, 51.826484018264836],
            [63.343328335832084, 59.3607305936073],
            [65.68215892053972, 47.945205479452056],
            [67.48125937031485, 51.82648401826485],
            [68.38080959520241, 59.58904109589041],
            [70.53973013493254, 48.630136986301395],
            [69.40029985007496, 41.7808219178082],
            [67.0014992503748, 42.69406392694063],
            [67.36131934032983, 35.61643835616438],
            [68.50074962518741, 27.397260273972606],
            [70.1799100449775, 29.680365296803615],
            [70.35982008995501, 29.680365296803615],
            [71.37931034482759, 34.70319634703196],
            [73.41829085457273, 34.47488584474884],
            [72.99850074962518, 25.799086757990878],
            [72.33883058470764, 24.657534246575324],
            [73.3583208395802, 18.721461187214587],
            [73.41829085457272, 17.579908675799103],
            [75.1574212893553, 18.036529680365277],
            [73.35832083958024, 12.785388127853864],
            [74.37781109445278, 13.013698630136972],
            [75.39730134932535, 12.78538812785385],
            [76.4167916041979, 8.67579908675799]
        ])
    
    # Find values to use to normalize data
    optim_length = data[np.argmax(data[:,1])][0] # length at peak force
    peak = np.max(data[:,1]) # peak force

    # Normalize each axis
    length = data[:,0]/optim_length
    force = data[:,1]/peak

    centres = np.arange(min(length)+0.1, max(length), .2)
    width = .15
    result = Regression(length, force, centres, width, .1, sigmoids=False) # Use Gaussian basis functions

    return result

force_length_regression = get_muscle_force_length_regression()
force_velocity_regression = get_muscle_force_velocity_regression()

def force_length_muscle(lm):
    """
    :param lm: muscle (contracile element) length
    :return: force-length scale factor
    """
    return force_length_regression.eval(lm)


def force_velocity_muscle(vm):
    """
    :param vm: muscle (contractile element) velocity)
    :return: force-velocity scale factor
    """
    return np.maximum(0, force_velocity_regression.eval(vm))

def plot_curves():
    """
    Plot force-length, force-velocity, SE, and PE curves.
    """
    lm = np.arange(0, 1.8, .01)
    vm = np.arange(-1.2, 1.2, .01)
    lt = np.arange(0, 1.07, .01)
    plt.subplot(2,1,1)
    plt.plot(lm, force_length_muscle(lm), 'r')
    plt.plot(lm, force_length_parallel(lm), 'g')
    plt.plot(lt, force_length_tendon(lt), 'b')
    plt.legend(('CE', 'PE', 'SE'))
    plt.xlabel('Normalized length')
    plt.ylabel('Force scale factor')
    plt.subplot(2, 1, 2)
    plt.plot(vm, force_velocity_muscle(vm), 'k')
    plt.xlabel('Normalized muscle velocity')
    plt.ylabel('Force Scale factor')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
        
#    plot_curves() # plot CE,SE,PE force-length curves and CE force-velocity curve
    print(get_velocity(1.0,np.array([1.0]),np.array([1.01]))) # calculate velocity given a=1.0,lm=1.0,ls=1.01

    # Constants
    resting_muscle_length = .3
    resting_tendon_length = .1
    max_isometric_force = 100.0
    total_length = resting_muscle_length + resting_tendon_length

    emg_data = load_data('./ta_vs_gait.csv')
    emg_data = np.array(emg_data)
    emg_data_regress = get_norm_emg(emg_data)
    
    frequency, duty_cycle, scaling, non_linearity = 35, 0.5, 1, -1
    a = Activation(frequency, duty_cycle, scaling, non_linearity)
    a.get_activation_signal(emg_data_regress)
    
    # Create an HillTypeMuscle using the given constants
    muscle = HillTypeMuscle(max_isometric_force, resting_muscle_length, resting_tendon_length)

    # Dynamic equation
    def f(t, x):
        normalized_tendon_length = muscle.norm_tendon_length(total_length,x)
        return get_velocity(a.get_amp(t), np.array([x]), np.array([normalized_tendon_length])) 

    # Simulate using rk45
    sol = solve_ivp(f, [0.6, 1], np.array([1.0]), max_step=.01, rtol=1e-5, atol=1e-8)
    a.plot()
    # Plot length and force over time
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(sol.t, sol.y.T*resting_muscle_length)
    plt.xlabel('Time (s)')
    plt.ylabel('CE length (m)')
    plt.subplot(2,1,2)
    plt.plot(sol.t, muscle.get_force(total_length,sol.y.T))
    plt.xlabel('Time (s)')
    plt.ylabel('Tension (N)')
    plt.tight_layout()
    plt.show()


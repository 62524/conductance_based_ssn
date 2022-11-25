from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
import multiprocess

class Dummy:
    pass

def naka(C, fmax, C_50, p=3.5):
    return fmax*C**p/(C_50**p + C**p)

def naka_inv(contrasts, fmax, C_50, p=3.5):
    return (C_50**p/fmax/(1/contrasts - 1/fmax))**(1/p)

def single_gaussian(theta, center, size_ext, periodic=True):
    if periodic: 
        return np.exp(-(theta - center)**2/(2*size_ext**2))+np.exp(-(theta - center+180)**2/(2*size_ext**2))+np.exp(-(theta - center-180)**2/(2*size_ext**2))
    else:
        return np.exp(-(theta - center)**2/(2*size_ext**2))

def average_firing_rate(step, dtime: second, spike_mon: SpikeMonitor):
    time_range = np.linspace(0, step*dtime/second, step+1, endpoint=True).reshape(-1, 1)*second
    train = spike_mon.spike_trains()
    avg_rate = np.array([np.sum(((time_range[:step] <= train[i])&(time_range[1:] > train[i])).astype(int), axis=1)/dtime for i in range(len(train))])
    return avg_rate.T

"""
def split_spike_train(step, dtime: second, spike_mon: SpikeMonitor):
    time_range = np.linspace(0, step*dtime/second, step+1, endpoint=True).reshape(-1, 1)*second
    train = spike_mon.spike_trains()
    return [(spike_mon.)[((time_range[:step] <= spike_mon.t)&(time_range[1:] > spike_mon.t))] for i in range(len(train))]
"""

def average_current(step, dtime, state: StateMonitor, name: str):
    time_range = np.linspace(0, step*dtime/second, step+1, endpoint=True).reshape(-1, 1)*second
    I = eval(f'state.{name}/pA')
    avg_current = np.array([np.average(I[:, (time_range[i] <= state.t)&(time_range[i+1] > state.t)], axis=1) for i in range(step)])
    return avg_current

def default_params():
    params = Dummy()
    init_network(params, 
    C=10,
    dtime=10*second,
    C_bg=18,
    N = 120,
    W = 1.028,
    N_input=1,
    center=45 ,
    size_ext=30,
    D=0, 
    sparsity=1, failure_rate=.5,
    sparsityEE=1,sparsityEI=1,sparsityIE=1,sparsityII=1,
    fmax=100.1,C_50=20,
    #Fixed Parameters
        tau_m=15*ms, tau_E=3*ms, tau_I=3*ms, tau_ref=3*ms,
        g_EI=1.42*nS, g_II=1.2*nS, g_EE=1.8*nS, g_IE= 2.06*nS, g_L=10*nS,
        R_L=-70*mV,  R_E=0*mV, R_I=-80*mV, V_th=-50*mV, V_r=-56*mV,
        A=0.2, B=0.8, sigma_ori=25,
    #Fixed Parameters
    eqs_neurons ="""
    dv/dt = (-(v - R_L) + (R_E - v)*g_E/g_L +  (R_I - v)*g_I/g_L + (R_E - v)*g_in/g_L)/tau_m : volt (unless refractory)
    dg_E/dt = -g_E/tau_E : siemens
    dg_I/dt = -g_I/tau_I : siemens
    dg_in/dt = (-g_in + avg_g_in)/tau_E: siemens
    avg_g_in = N_input*r_ext(t, i)*tau_E*g_ext : siemens
    I_exc = g_E*(R_E - v): ampere
    I_inh = abs(g_I*(R_I - v)): ampere
    I_ext = g_in*(R_E - v): ampere
    I_net = g_I*(R_I - v) + g_in*(R_E - v) + g_E*(R_E - v) : ampere
    theta : 1
    """,
    w_eqs = "w = int(abs(theta_pre-theta_post)<=90)*(A + B*exp(-(abs(theta_pre-theta_post)/(sqrt(2)*sigma_ori))**2)) + int(abs(theta_pre-theta_post)>90)*(A + B*exp(-((180-abs(theta_pre-theta_post))/(sqrt(2)*sigma_ori))**2)): 1 "
    )
    return params


def init_network(params, input_shape="single_gaussian", time_dependency=None, cpp_standalone = True, **kwargs):
    start_scope()
    # 使用cpp standalone
    if cpp_standalone:
        device.reinit()
        device.activate()
        pid = os.getpid()
        directory = f"standalone{pid}"
        set_device('cpp_standalone', directory=directory)

    for key, value in kwargs.items():
        params.__dict__.update({key: value})
    
    params.sparsityEE = params.sparsityEE
    params.sparsityIE = params.sparsityIE
    params.sparsityEI = params.sparsityEI
    params.sparsityII = params.sparsityII
    params.theta = np.concatenate((np.linspace(1, 180, params.N), np.linspace(1, 180, params.N)))
    params.g_ext = 10/params.N_input*nS
    #params.contrast = naka(params.C, params.fmax, params.C_50)
    if input_shape=="single_gaussian":
        params.r_ext = TimedArray((params.C)*single_gaussian(params.theta, params.center, params.size_ext).reshape((-1, 2*params.N))*Hz, dt=params.dtime)
    if time_dependency=="contrast":
        params.r_ext = TimedArray((params.Cs.reshape(-1, 1))*single_gaussian(params.theta, params.center, params.size_ext).reshape((-1, 2*params.N))*Hz, dt=params.dtime)
    if time_dependency=="width":
        params.r_ext = TimedArray(params.C*single_gaussian(params.theta, params.center, params.widths.reshape((-1, 1)))*Hz, dt=params.dtime)
    if time_dependency=="double_gaussian_ratio":
        params.r_ext = TimedArray((params.C_total*(1-params.ratios.reshape(-1, 1))*single_gaussian(params.theta, params.center, params.size_ext).reshape((1, -1)) + params.C_total*params.ratios.reshape(-1, 1)*single_gaussian(params.theta, params.center2, params.size_ext).reshape((1, -1)))*Hz, dt=params.dtime)
    if time_dependency=="contrast_double_gaussian":
        params.r_ext = TimedArray((params.Cs.reshape(-1, 1))*(single_gaussian(params.theta, params.center, params.size_ext)+single_gaussian(params.theta, params.center+90, params.size_ext)).reshape(1, -1)*Hz, dt=params.dtime)
    params.neurons = NeuronGroup(2*params.N, model=params.eqs_neurons, threshold='v>V_th', reset='v=V_r', refractory=params.tau_ref, method='euler', name="neurons")
    params.g_input = PoissonGroup(2*params.N, rates="N_input*r_ext(t, i)", name="poisson")    # Poission Input External Neurons
    params.input_to_G = Synapses(params.g_input, params.neurons, on_pre='g_in += g_ext', name="input_to_G") # Setting firing condition,
    params.g_background = PoissonGroup(2*params.N, rates="N_input*C_bg*Hz", name='background')
    params.background_to_G = Synapses(params.g_background, params.neurons, on_pre='g_in += g_ext', name="background_to_G") # Setting firing condition,
    params.E_neurons = params.neurons[:params.N]    # Excitatory Neurons
    params.I_neurons = params.neurons[params.N:]    # Inhibitory Neurons 
    params.E_to_E = Synapses(params.E_neurons, params.E_neurons, model=params.w_eqs, on_pre='g_E+=W*w*g_EE', name="E_to_E")
    params.I_to_E = Synapses(params.I_neurons, params.E_neurons, model=params.w_eqs, on_pre='g_I+=W*w*g_EI', name="I_to_E")
    params.E_to_I = Synapses(params.E_neurons, params.I_neurons, model=params.w_eqs, on_pre='g_E+=W*w*g_IE', name="E_to_I")
    params.I_to_I = Synapses(params.I_neurons, params.I_neurons, model=params.w_eqs, on_pre='g_I+=W*w*g_II', name="I_to_I")

def build_network(params):
    params.neurons.v = params.V_r
    params.neurons.theta = params.theta
    params.E_to_E.connect(p=params.sparsityEE), params.I_to_E.connect(p=params.sparsityIE), params.E_to_I.connect(p=params.sparsityEI), params.I_to_I.connect(p=params.sparsityII), params.input_to_G.connect(j='i'), params.background_to_G.connect(j='i')
    params.spike_E = SpikeMonitor(params.neurons[:params.N])
    params.spike_I = SpikeMonitor(params.neurons[params.N:])
    network = Network(name='network')
    network.add([params.neurons,params.g_input,params.input_to_G,params.E_to_E,params.I_to_E,params.E_to_I,params.I_to_I])
    if params.C_bg>0: network.add([params.g_background, params.background_to_G])
    network.add([params.spike_E,params.spike_I])
    params.network = network

def run_network(params, duration: second, variables=['I_exc', 'I_inh', 'I_ext', 'I_net', 'g_in', 'avg_g_in', 'g_E', 'g_I']):

    params.state_E = StateMonitor(params.E_neurons, variables=variables, record=True, dt=20*ms)
    params.state_I = StateMonitor(params.I_neurons, variables=variables, record=True, dt=20*ms)
    params.network.add([params.state_E,params.state_I])
    params.network.run(duration, namespace=params.__dict__)


def dict_diff(first, second):
    """
    比较两个字典的不同
    """
    KEYNOTFOUNDIN1 = '<KEYNOTFOUNDIN1>'  # 没在第一个字典中找到
    KEYNOTFOUNDIN2 = '<KEYNOTFOUNDIN2>'  # 没在第二个字典中找到
    
    diff = {}
    sd1 = set(first)
    sd2 = set(second)
    # 第二个字典中不存在的键
    for key in sd1.difference(sd2):
        diff[key] = KEYNOTFOUNDIN2
    # 第一个字典中不存在的键
    for key in sd2.difference(sd1):
        diff[key] = KEYNOTFOUNDIN1
    # 比较不同
    for key in sd1.intersection(sd2):
        if key in ['theta','r_ext','network','E_neurons','E_to_E','E_to_I','I_neurons','I_to_E','I_to_I','background_to_G','g_background','g_input','input_to_G','neurons','spike_I', 'spike_E', 'network']:
            continue
        if first[key] != second[key]:
            diff[key] = (first[key], second[key])
    return diff

def save_params(params,name = 'default'):
    default_param = default_params()
    build_network(default_param)
    default_dict = default_param.__dict__
    diff = dict_diff(params.__dict__,default_dict,)
    now = datetime.now().strftime("%m_%d %H_%M_%S")
    with open('params'+now+' '+name+'.txt', 'w') as file:
        file.write(str(default_dict)+'\n')
        file.write("\n")
        file.write("defferent params:")
        file.write(str(diff))



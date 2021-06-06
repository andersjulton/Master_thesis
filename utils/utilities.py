import numpy as np
from qutip import qzero, qeye, sigmax, sigmay, sigmaz
import sys

def query_yes_no(question, default="yes"):
    """
    Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def print_action(env, t=0, action=0):

    actions = env.actions
    t = str(t+1)
    if actions[action] == qeye(2):
        print("Pulse "+ t + ": Identity operator")
    elif actions[action] == 1j*sigmax():
        print("Pulse "+ t + ": x-rotation")
    elif actions[action] == 1j*sigmay():
        print("Pulse "+ t + ": y-rotation")
    elif actions[action] == 1j*sigmaz():
        print("Pulse "+ t + ": z-rotation")

def get_action_string(env, action):
    actions = env.actions
    if actions[action] == qeye(2):
        return "I"
    elif actions[action] == 1j*sigmax():
        return "X"
    elif actions[action] == 1j*sigmay():
        return "Y"
    elif actions[action] == 1j*sigmaz():
        return "Z"

def Rx(theta, QT=True):
    rx = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],[-1j*np.sin(theta/2),np.cos(theta/2)]])
    if QT:
        rx += qzero(2)
    return rx

def Ry(theta, QT=True):
    ry = np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2),np.cos(theta/2)]])
    if QT:
        ry += qzero(2)
    return ry

def Rz(theta, QT=True):
    rz = np.array([[np.exp(-1j*theta/2), 0],[0, np.exp(1j*theta/2)]])
    if QT:
        rz += qzero(2)
    return rz

"""Class containing equations."""

import cupy as cp

def test_training_data(train):
    """Calculate the travel time between two points including wait time at the end node."""
    # copy and convert our pandas dataframe into numpy variables.
    # I will leave the target at Y, as I can't see where to set that in the code, but it's in there somehow!

    # NOTE: I'm only feeding in the TRAIN values to the algorithms. Later I will independely check
    # the MSE myself using a holdout test dataset

    A = train.Attack.values
    AL  = train.Attacker_Level.values
    P = train.Power.values
    D = train.Defense.values
    DL = train.Defender_Level.values
    H = train.Health.values

    R = train.Ratio.values  # this is our target, now mapped to Y
    return [A, AL, P, D, DL, H], R

@staticmethod
def attacker_value_training_data(train):
    """Calculate the travel time between two points including wait time at the end node."""
    # copy and convert our pandas dataframe into numpy variables.
    # I will leave the target at Y, as I can't see where to set that in the code, but it's in there somehow!

    # NOTE: I'm only feeding in the TRAIN values to the algorithms. Later I will independely check
    # the MSE myself using a holdout test dataset

    A = train.Attack.values
    AL  = train.Attacker_Level.values

    AV = train.Attacker_Value.values  # this is our target, now mapped to Y
    return [A, AL], AV

@staticmethod
def attacker_defender_value_training_data(train):
    """Calculate the travel time between two points including wait time at the end node."""
    # copy and convert our pandas dataframe into numpy variables.
    # I will leave the target at Y, as I can't see where to set that in the code, but it's in there somehow!

    # NOTE: I'm only feeding in the TRAIN values to the algorithms. Later I will independely check
    # the MSE myself using a holdout test dataset

    A = train.Attack.values
    AL  = train.Attacker_Level.values
    D = train.Defense.values
    DL = train.Defender_Level.values

    ADV = train.Attacker_Defender_Value.values  # this is our target, now mapped to Y
    return [A, AL, D, DL], ADV

@staticmethod
def damage_training_data(train):
    """Calculate the travel time between two points including wait time at the end node."""
    # copy and convert our pandas dataframe into numpy variables.
    # I will leave the target at Y, as I can't see where to set that in the code, but it's in there somehow!

    # NOTE: I'm only feeding in the TRAIN values to the algorithms. Later I will independely check
    # the MSE myself using a holdout test dataset

    P = train.Attack.values
    AL  = train.Attacker_Level.values
    P = train.Power.values
    D = train.Defense.values
    DL = train.Defender_Level.values

    R = train.Damage.values  # this is our target, now mapped to Y
    return [P, AL, P, D, DL], R

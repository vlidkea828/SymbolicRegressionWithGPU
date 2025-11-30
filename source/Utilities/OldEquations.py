"""Class containing the structure of a node graph."""
from math import floor

class OldEquationSolver:
    """Defenition of a node graph."""
    def __init__(
            self,
            attack : float,
            attacker_level : float,
            power : float,
            defense : float,
            defender_level : float,
            health : float
    ):
        self.attack = attack
        self.attacker_level = attacker_level
        self.power = power
        self.defense = defense
        self.defender_level = defender_level
        self.health = health

    def calculate_stat(
            self,
            baseStat : float,
            level : float
    ):
        """Insert new node into graph."""
        return floor((2 * baseStat * level) / 100) + 5
    
    def calculate_attacker_level_contribution(
            self,
            level : float
    ):
        """Insert new node into graph."""
        return (((2 * level) / 5) + 2) / 50
    
    def calculate_hp(
            self,
            baseStat : float,
            level : float
    ):
        """Insert new node into graph."""
        return floor((2 * baseStat * level) / 100) + level + 10
    
    def evaluate_stat_difference(
            self,
    ):
        """Insert new node into graph."""
        return self.calculate_stat(self.attack, self.attacker_level) \
            / self.calculate_stat(self.defense, self.defender_level)
    
    def evaluate_damage(
            self,
    ):
        """Insert new node into graph."""
        return round((self.power * \
            self.calculate_attacker_level_contribution(self.attacker_level) \
            * self.evaluate_stat_difference()) + 2)

    def evaluate_remaining_hp(
            self,
    ):
        """Insert new node into graph."""
        return max(self.calculate_hp(self.health, self.defender_level) - \
                self.evaluate_damage(), 0)
    
    def evaluate_ratio(
            self,
    ):
        """Insert new node into graph."""
        return self.evaluate_remaining_hp() \
            / self.calculate_hp(self.health, self.defender_level)
    
    def evaluate_attacker_value(
            self
    ):
        """Evaluate the current values attacker value"""
        return self.calculate_attacker_level_contribution(self.attacker_level) \
                * self.calculate_stat(self.attack, self.attacker_level)
    
    def evaluate_attacker_defender_value(
            self
    ):
        """Evaluate the current values attacker defender value"""
        return self.calculate_attacker_level_contribution(self.attacker_level) \
                * self.evaluate_stat_difference()

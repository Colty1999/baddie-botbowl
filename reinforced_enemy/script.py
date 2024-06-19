class MyScriptedBot(ProcBot):

    def __init__(self, name):
        super().__init__(name)
        self.my_team = None
        self.opp_team = None
        self.actions = []
        self.last_turn = 0
        self.last_half = 0

        self.off_formation = [
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "m", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "x", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "S"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "x"],
            ["-", "-", "-", "-", "-", "s", "-", "-", "-", "0", "-", "-", "S"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "x"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "S"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "x", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "m", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
        ]

        self.def_formation = [
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "x", "-", "b", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "x", "-", "S", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "0"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "0"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "0"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "x", "-", "S", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "x", "-", "b", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
        ]

        self.off_formation = Formation("Wedge offense", self.off_formation)
        self.def_formation = Formation("Zone defense", self.def_formation)
        self.setup_actions = []


    def coin_toss_flip(self, game):
        """
        Select heads/tails and/or kick/receive
        """
        return Action(ActionType.TAILS)
        # return Action(ActionType.HEADS)

    def coin_toss_kick_receive(self, game):
        """
        Select heads/tails and/or kick/receive
        """
        return Action(ActionType.RECEIVE)
        # return Action(ActionType.KICK)

    def setup(self, game):
        """
        Use either a Wedge offensive formation or zone defensive formation.
        """
        # Update teams
        self.my_team = game.get_team_by_id(self.my_team.team_id)
        self.opp_team = game.get_opp_team(self.my_team)

        if self.setup_actions:
            action = self.setup_actions.pop(0)
            return action

        # If traditional board size
        if game.arena.width == 28 and game.arena.height == 17:
            if game.get_receiving_team() == self.my_team:
                self.setup_actions = self.off_formation.actions(game, self.my_team)
                self.setup_actions.append(Action(ActionType.END_SETUP))
            else:
                self.setup_actions = self.def_formation.actions(game, self.my_team)
                self.setup_actions.append(Action(ActionType.END_SETUP))
            action = self.setup_actions.pop(0)
            return action

        # If smaller variant - use built-in setup actions

        for action_choice in game.get_available_actions():
            if action_choice.action_type != ActionType.END_SETUP and action_choice.action_type != ActionType.PLACE_PLAYER:
                self.setup_actions.append(Action(ActionType.END_SETUP))
                return Action(action_choice.action_type)

        # This should never happen
        return None
    

    def place_ball(self, game):
        """
        Place the ball when kicking.
        """
        side_width = game.arena.width / 2
        side_height = game.arena.height
        squares_from_left = math.ceil(side_width / 2)
        squares_from_right = math.ceil(side_width / 2)
        squares_from_top = math.floor(side_height / 2)
        left_center = Square(squares_from_left, squares_from_top)
        right_center = Square(game.arena.width - 1 - squares_from_right, squares_from_top)
        if game.is_team_side(left_center, self.opp_team):
            return Action(ActionType.PLACE_BALL, position=left_center)
        return Action(ActionType.PLACE_BALL, position=right_center)
    

    def reroll(self, game): #dodaj sprawdzenie czy dana akcja zakonczy ture i jesli tak rerolluj
        """
        Select between USE_REROLL and DONT_USE_REROLL
        """
        reroll_proc = game.get_procedure()
        context = reroll_proc.context
        is_safe = 0.5
        if type(context) == botbowl.Dodge:
            success_chance = self.calculate_dodge_success(game)
            if is_safe < success_chance:
                return Action(ActionType.USE_REROLL)
            else:
                return Action(ActionType.DONT_USE_REROLL)
        if type(context) == botbowl.Pickup:
            return Action(ActionType.USE_REROLL)
        if type(context) == botbowl.PassAttempt:
            return Action(ActionType.USE_REROLL)
        if type(context) == botbowl.Catch:
            return Action(ActionType.USE_REROLL)
        if type(context) == botbowl.GFI:
            return Action(ActionType.USE_REROLL)
        if type(context) == botbowl.BloodLust:
            return Action(ActionType.USE_REROLL)
        if type(context) == botbowl.Block:
            attacker = context.attacker
            attackers_down = 0
            for die in context.roll.dice:
                if die.get_value() == BBDieResult.ATTACKER_DOWN:
                    attackers_down += 1
                elif die.get_value() == BBDieResult.BOTH_DOWN and not attacker.has_skill(Skill.BLOCK) and not attacker.has_skill(Skill.WRESTLE):
                    attackers_down += 1
            if attackers_down > 0 and context.favor != self.my_team:
                return Action(ActionType.USE_REROLL)
            if attackers_down == len(context.roll.dice) and context.favor != self.opp_team:
                return Action(ActionType.USE_REROLL)
            return Action(ActionType.DONT_USE_REROLL)
        return Action(ActionType.DONT_USE_REROLL)

    def calculate_dodge_success(self,game):
        ag = game.player.get_agility()
        required_roll = 7 - ag  # Zakładamy prosty modyfikator 7 - czyli bez zadnych udziwnnien
        success_chance = (6 - required_roll + 1) / 6.0
        return  success_chance
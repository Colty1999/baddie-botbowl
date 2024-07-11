from botbowl import OutcomeType, Game
from Data.scripted_bot import ScriptedBot
from botbowl import OutcomeType, Game
import botbowl.core.procedure as procedure


class A2C_Reward:
    # --- Reward function ---
    rewards_own = {
        OutcomeType.TOUCHDOWN: 5.0,  #Divided all values by 2
        # OutcomeType.SUCCESSFUL_CATCH: 0.25,
        # OutcomeType.INTERCEPTION: 0.05,
        # OutcomeType.SUCCESSFUL_PICKUP: 0.125,
        # OutcomeType.FUMBLE: -0.05,
        # OutcomeType.KNOCKED_DOWN: -0.075,
        # OutcomeType.KNOCKED_OUT: -0.10,
        # OutcomeType.CASUALTY: -0.15
        OutcomeType.SUCCESSFUL_CATCH: 0.025,
        OutcomeType.INTERCEPTION: 0.05,
        OutcomeType.SUCCESSFUL_PICKUP: 0.2,
        OutcomeType.FUMBLE: -0.05,
        OutcomeType.KNOCKED_DOWN: -0.15,
        OutcomeType.KNOCKED_OUT: -0.2,
        OutcomeType.CASUALTY: -0.25,
        OutcomeType.FAILED_GFI: -0.05,
        OutcomeType.FAILED_DODGE: -0.05,
        OutcomeType.END_OF_GAME_WINNER: 0.5  #todo check if this will help
        # OutcomeType.TURNOVER: -1.2
    }
    rewards_opp = {
        # OutcomeType.TOUCHDOWN: -1,
        # OutcomeType.SUCCESSFUL_CATCH: -0.05,
        # OutcomeType.INTERCEPTION: -0.1,
        # OutcomeType.SUCCESSFUL_PICKUP: -0.15,
        # OutcomeType.FUMBLE: 0.025,
        # OutcomeType.KNOCKED_DOWN: 0.05,
        # OutcomeType.KNOCKED_OUT: 0.075,
        # OutcomeType.CASUALTY: 0.15
        OutcomeType.TOUCHDOWN: -5.0,
        OutcomeType.SUCCESSFUL_CATCH: -0.05,
        OutcomeType.INTERCEPTION: -0.1,
        OutcomeType.SUCCESSFUL_PICKUP: -0.4,
        OutcomeType.FUMBLE: 0.05,
        OutcomeType.KNOCKED_DOWN: 0.25,
        OutcomeType.KNOCKED_OUT: 0.35,
        OutcomeType.CASUALTY: 0.45,
        OutcomeType.FAILED_GFI: 0.05,
        OutcomeType.FAILED_DODGE: 0.05,
        # OutcomeType.END_OF_GAME_WINNER: -1
        # OutcomeType.TURNOVER: 1.2

    }
    ball_progression_reward = 0.005

    def __init__(self):
        self.last_report_idx = 0
        self.last_ball_x = None
        self.last_ball_team = None

    def __call__(self, game: Game):
        if len(game.state.reports) < self.last_report_idx:
            self.last_report_idx = 0

        r = 0.0
        own_team = game.active_team
        opp_team = game.get_opp_team(own_team)

        for outcome in game.state.reports[self.last_report_idx:]:
            team = None
            if outcome.player is not None:
                team = outcome.player.team
            elif outcome.team is not None:
                team = outcome.team
            if team == own_team and outcome.outcome_type in A2C_Reward.rewards_own:
                r += A2C_Reward.rewards_own[outcome.outcome_type]
            if team == opp_team and outcome.outcome_type in A2C_Reward.rewards_opp:
                r += A2C_Reward.rewards_opp[outcome.outcome_type]

        self.last_report_idx = len(game.state.reports)

        ball_carrier = game.get_ball_carrier()
        if ball_carrier is not None:
            if self.last_ball_team is own_team and ball_carrier.team is own_team:
                ball_progress = self.last_ball_x - ball_carrier.position.x
                if own_team is game.state.away_team:
                    ball_progress *= -1  # End zone at max x coordinate
                r += A2C_Reward.ball_progression_reward * ball_progress

            self.last_ball_team = ball_carrier.team
            self.last_ball_x = ball_carrier.position.x
        else:
            self.last_ball_team = None
            self.last_ball_x = None

        return r


def a2c_scripted_actions(game: Game):
    proc_type = type(game.get_procedure())
    if proc_type is procedure.Block:
        # noinspection PyTypeChecker
        return ScriptedBot.block(self=None, game=game)
    if proc_type is procedure.CoinTossFlip:
        return ScriptedBot.coin_toss_flip(self=None, game=game)
    if proc_type is procedure.CoinTossKickReceive:
        return ScriptedBot.coin_toss_kick_receive(self=None, game=game)
    if proc_type is procedure.PlaceBall:
        return ScriptedBot.place_ball(self=None, game=game)
    if proc_type is procedure.Reroll:
        return ScriptedBot.reroll(self=None, game=game)
    if proc_type is procedure.Setup:
        return ScriptedBot.setup(game=game)
    return None

"""Standard basketball formulas, in one place so each can be tested by hand.

Definitions follow Basketball-Reference's glossary, which uses Dean
Oliver's work. Anything that is an estimate rather than a count is
named as one.
"""
from ..util import safe_div


def estimate_possessions(*, fga, fta, oreb, tov, fgm, opp_dreb,
                         opp_fga, opp_fta, opp_oreb, opp_tov, opp_fgm, dreb) -> float:
    """Box-score possession estimate for one game (same value for both teams).

    Averages each team's estimate:
      FGA + 0.4*FTA - 1.07 * (ORB / (ORB + Opp DRB)) * (FGA - FGM) + TOV
    This is an estimate: exact counts need play-by-play (Phase 2).
    """
    def one(fga_, fta_, oreb_, tov_, fgm_, opp_dreb_):
        reb_chances = oreb_ + opp_dreb_
        orb_share = oreb_ / reb_chances if reb_chances else 0.0
        return fga_ + 0.4 * fta_ - 1.07 * orb_share * (fga_ - fgm_) + tov_

    return 0.5 * (one(fga, fta, oreb, tov, fgm, opp_dreb)
                  + one(opp_fga, opp_fta, opp_oreb, opp_tov, opp_fgm, dreb))


def pace(possessions, team_minutes):
    """Possessions per 48 minutes. team_minutes is player-minutes (240 regulation)."""
    return safe_div(48 * possessions, team_minutes / 5 if team_minutes else None)


def rating(points, possessions):
    """Points per 100 possessions (offensive or defensive rating)."""
    return safe_div(100 * points, possessions)


def true_shooting(pts, fga, fta):
    """TS% = PTS / (2 * (FGA + 0.44 * FTA))."""
    return safe_div(pts, 2 * (fga + 0.44 * fta))


def effective_fg(fgm, fg3m, fga):
    """eFG% = (FGM + 0.5 * 3PM) / FGA."""
    return safe_div(fgm + 0.5 * fg3m, fga)


def turnover_rate(tov, fga, fta):
    """TOV% = TOV / (FGA + 0.44 * FTA + TOV)."""
    return safe_div(tov, fga + 0.44 * fta + tov)


def oreb_rate(oreb, opp_dreb):
    """ORB% = ORB / (ORB + Opp DRB)."""
    return safe_div(oreb, oreb + opp_dreb)


def ft_rate(fta, fga):
    """Free throw rate = FTA / FGA."""
    return safe_div(fta, fga)


def three_rate(fg3a, fga):
    """3-point attempt rate = 3PA / FGA."""
    return safe_div(fg3a, fga)


# -- player rate stats ------------------------------------------------------
# Each is split into a numerator and a denominator for a single game, so a
# window of games can be combined exactly: rate = sum(num) / sum(den).


def usage_parts(*, minutes, fga, fta, tov, tm_minutes, tm_fga, tm_fta, tm_tov):
    """USG% = 100 * (FGA + 0.44*FTA + TOV) * (TmMP/5) / (MP * (TmFGA + 0.44*TmFTA + TmTOV))."""
    num = fga + 0.44 * fta + tov
    den = (minutes * (tm_fga + 0.44 * tm_fta + tm_tov) / (tm_minutes / 5)) if tm_minutes else 0.0
    return num, den


def assist_pct_parts(*, minutes, ast, fgm, tm_minutes, tm_fgm):
    """AST% = 100 * AST / ((MP / (TmMP/5)) * TmFGM - FGM): share of teammate baskets assisted."""
    den = (minutes / (tm_minutes / 5)) * tm_fgm - fgm if tm_minutes else 0.0
    return ast, max(den, 0.0)


def rebound_pct_parts(*, minutes, reb, tm_minutes, tm_reb, opp_reb):
    """REB% = 100 * REB * (TmMP/5) / (MP * (TmREB + OppREB)): share of available rebounds."""
    den = (minutes * (tm_reb + opp_reb) / (tm_minutes / 5)) if tm_minutes else 0.0
    return reb, den


def pythagorean_win_pct(points_for, points_against, exponent=14.0):
    """Expected win share from points scored and allowed (Morey's NBA exponent ~13.9-14)."""
    if not points_for and not points_against:
        return None
    pf, pa = points_for ** exponent, points_against ** exponent
    return pf / (pf + pa)

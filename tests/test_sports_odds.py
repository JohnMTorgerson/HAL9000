"""ESPN data projection, without live sports requests."""
from unittest.mock import patch

import pytest

from sports_api_espnnfl import ESPNnflAPI


@pytest.fixture
def event():
    return {'id': '12345', 'date': '2026-10-11T17:00Z', 'competitions': [{
        'competitors': [{'homeAway': 'home', 'team': {'displayName': 'New Orleans Saints'}, 'score': '0'},
                        {'homeAway': 'away', 'team': {'displayName': 'Minnesota Vikings'}, 'score': '0'}],
        'status': {'type': {'name': 'STATUS_SCHEDULED', 'completed': False}},
        'odds': [{'provider': {'name': 'Fixture Bookmaker'}, 'details': 'MIN -2.5', 'overUnder': 41.5,
                  'moneyline': {'home': {'close': {'odds': '+120', 'link': {'secret': 'discard'}}},
                                'away': {'close': {'odds': '-142'}, 'open': {'odds': '-120'}}},
                  'pointSpread': {'away': {'close': {'line': '-2.5', 'odds': '-108'},
                                           'open': {'line': '-1.5', 'odds': '-105'}}},
                  'total': {'over': {'close': {'line': 'o41.5', 'odds': '-108'}}}}]}]}


def test_find_game_passes_odds_with_sides_source_and_opening_separate(event):
    api = ESPNnflAPI()
    with patch.object(api, '_fetch_full_season', return_value=[event]):
        games = api.find_game('Minnesota Vikings', 'New Orleans Saints')
    game = games[0]
    odds = game['odds']
    book = odds['bookmakers'][0]
    assert game['home_team'] == 'New Orleans Saints'
    assert odds['status'] == 'available' and odds['line_type'] == 'listed'
    assert odds['retrieved_at'] and odds['source'] == 'ESPN'
    assert odds['url'].endswith('/12345')
    assert book['bookmaker'] == 'Fixture Bookmaker'
    assert book['spread_description'] == 'MIN -2.5'
    assert book['listed']['pointSpread']['away']['line'] == '-2.5'
    assert book['opening']['pointSpread']['away']['line'] == '-1.5'
    assert book['listed']['moneyline']['home'] == {'odds': '+120'}
    assert book['opening']['moneyline']['away'] == {'odds': '-120'}
    assert book['total_points'] == 41.5


@pytest.mark.parametrize('value', [None, [], {}, [None], [{'provider': {'name': 'Book'}}]])
def test_missing_or_partial_odds_are_not_invented(event, value):
    event['competitions'][0]['odds'] = value
    result = ESPNnflAPI()._parse_event(event)
    assert result['odds']['status'] == 'not_available'
    assert result['odds']['bookmakers'] == []
    assert result['status'] == 'STATUS_SCHEDULED'


def test_legacy_team_moneylines_and_no_derived_spread_sign(event):
    event['competitions'][0]['odds'] = [{'provider': {'name': 'Older Feed'}, 'spread': 2.5,
        'details': 'MIN -2.5', 'overUnder': 41.5,
        'homeTeamOdds': {'moneyLine': 120}, 'awayTeamOdds': {'moneyLine': -142}}]
    odds = ESPNnflAPI()._parse_event(event)['odds']['bookmakers'][0]
    assert odds['listed']['moneyline'] == {'home': {'odds': 120}, 'away': {'odds': -142}}
    assert 'pointSpread' not in odds['listed']
    assert odds['spread_description'] == 'MIN -2.5'


def test_completed_game_marks_lines_as_closing(event):
    event['competitions'][0]['status']['type'].update(name='STATUS_FINAL', completed=True)
    assert ESPNnflAPI()._parse_event(event)['odds']['line_type'] == 'closing'

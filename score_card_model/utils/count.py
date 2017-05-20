__all__ = ['count_binary']

from score_card_model.utils.check import check_array_binary


def count_binary(a, event=1):
    if not check_array_binary(a):
        raise AttributeError("array must be a binary array")
    try:
        event_count = (a == event).sum()
    except AttributeError as ae:
        raise AttributeError("need a event")
    non_event_count = a.shape[-1] - event_count
    return event_count, non_event_count

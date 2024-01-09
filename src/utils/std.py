from typing import Any, Dict, List


def merge_dicts_of_lists(d1: Dict[str, list], d2: Dict[str, Any]) -> Dict[str, list]:
    for k, v in d2.items():
        d1[k].append(v)
    return d1


def flatten(lis: List[List[Any]]) -> List[Any]:
    return [la for li in lis for la in li]

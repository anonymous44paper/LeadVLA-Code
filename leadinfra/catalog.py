"""Portable asset registration and compositional appearance compatibility."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .compose import semantic_id

CATEGORIES = {"environment", "route", "robot", "human", "sensor", "object"}


@dataclass(frozen=True)
class Asset:
    identifier: str
    category: str
    relative_file: str

    def __post_init__(self):
        semantic_id(self.identifier)
        if self.category not in CATEGORIES:
            raise ValueError("Unknown asset category")
        path = PurePosixPath(self.relative_file)
        if not self.relative_file or path.is_absolute() or ".." in path.parts or "\\" in self.relative_file or ":" in self.relative_file:
            raise ValueError("Asset files must be portable relative paths")


class Catalog:
    def __init__(self, assets):
        self.assets = {}
        for asset in assets:
            if asset.identifier in self.assets:
                raise ValueError("Duplicate asset identifier")
            self.assets[asset.identifier] = asset

    def resolve(self, identifier, asset_root):
        root = Path(asset_root).resolve()
        resolved = (root / self.assets[identifier].relative_file).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("Resolved asset escapes the catalog root")
        if not resolved.is_file():
            raise FileNotFoundError(f"Asset is not installed: {identifier}")
        return resolved

    def inventory(self):
        return {category: sum(a.category == category for a in self.assets.values()) for category in sorted(CATEGORIES)}


def appearance_combinations(parts, compatible):
    """Yield caller-validated body/clothing/accessory/scale combinations.

    Actual meshes and compatibility tables must come from an installed catalog.
    """
    from itertools import product
    required = ("body", "clothing", "accessory", "scale")
    if set(parts) != set(required) or any(not parts[key] for key in required):
        raise ValueError("Provide nonempty body, clothing, accessory and scale catalogs")
    for values in product(*(parts[key] for key in required)):
        candidate = dict(zip(required, values))
        for value in values:
            semantic_id(value)
        if compatible(candidate):
            yield candidate

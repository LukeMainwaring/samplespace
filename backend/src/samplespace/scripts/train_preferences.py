"""Manually train the preference model from accumulated verdicts.

Usage:
    uv run --directory backend train-preferences
"""

import asyncio
import logging

from samplespace.dependencies.db import get_async_sqlalchemy_session
from samplespace.services import preference as preference_service
from samplespace.services.preference import FEATURE_DISPLAY_NAMES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def run_training() -> None:
    async with get_async_sqlalchemy_session() as db:
        meta = await preference_service.train(db)

    if meta is None:
        logger.info("Not enough data to train. Collect more pair verdicts with both approvals and rejections.")
        return

    logger.info(f"Trained preference model v{meta.version}")
    logger.info(f"  Accuracy: {meta.accuracy:.1%} (cross-validated)")
    logger.info(f"  Verdicts: {meta.verdict_count}")
    logger.info("  Feature importances:")
    for name, importance in sorted(meta.feature_importances.items(), key=lambda x: -x[1]):
        display = FEATURE_DISPLAY_NAMES.get(name, name)
        logger.info(f"    {display}: {importance:.1%}")


def main() -> None:
    asyncio.run(run_training())


if __name__ == "__main__":
    main()

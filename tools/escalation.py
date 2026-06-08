"""
Escape hatches for unsolvable situations:

- StuckDetector: flags when the page stops changing across repeated actions
  (e.g. a button that does nothing, a captcha wall), so the agent escalates
  instead of looping forever.
- ask_human: surface a question and pause for the human (same idea as the
  classifier's UNCERTAIN flow).
- abort_workflow: give up gracefully, closing the browser session.
"""

from .browser import close_session


class StuckDetector:
    """Counts consecutive identical page signatures; trips after `threshold`."""

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self._last: str | None = None
        self._count = 0

    def record(self, signature: str) -> bool:
        """Record a page signature. Returns True if we appear stuck."""
        if signature == self._last:
            self._count += 1
        else:
            self._last = signature
            self._count = 1
        return self._count >= self.threshold

    def reset(self) -> None:
        self._last = None
        self._count = 0


def ask_human(question: str) -> dict:
    """
    Escalate a question to the human and pause the workflow.

    Use when stuck on something only the human can resolve: a captcha, a 2FA
    code, a login failure, or an unrecognized page where the right action is
    unclear.

    Args:
        question: A single specific question for the human.

    Returns:
        A signal dict; after calling this, stop and present the question to the
        user, then wait for their reply before continuing.
    """
    return {"status": "needs_human", "question": question}


async def abort_workflow(reason: str) -> str:
    """
    Abort the current browsing workflow and close the browser cleanly.

    Use when the task can't be completed and isn't worth escalating (e.g. the
    site is down, or the offer is gone).

    Args:
        reason: Why the workflow is being aborted.

    Returns:
        A confirmation string.
    """
    await close_session()
    return f"Workflow aborted: {reason}"

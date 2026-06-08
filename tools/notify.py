from rich.console import Console

console = Console()


def notify_moneymaker(title: str, url: str, reasoning: str, estimated_value: float | None = None) -> str:
    """
    Notify about a MONEYMAKER post. Currently prints to console; extend later for email.

    Args:
        title: Post title
        url: Post URL
        reasoning: Why this was classified as a moneymaker
        estimated_value: Estimated dollar value of the opportunity, if known

    Returns:
        Confirmation string.
    """
    value_str = f" (~${estimated_value:.0f})" if estimated_value else ""
    console.print(f"\n[bold green]MONEYMAKER{value_str}[/bold green]: {title}")
    console.print(f"  [dim]{url}[/dim]")
    console.print(f"  {reasoning}")
    return f"Notified: {title}"

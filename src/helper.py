"""
===============================================================================
                            CineRate AI
                        Professional Helper Module
-------------------------------------------------------------------------------
Author      : Anusha G Talawar
Project     : Movie Rating Prediction using Machine Learning
===============================================================================
"""

from datetime import datetime


# =============================================================================
# Rating Badge
# =============================================================================

def rating_badge(rating):

    rating = float(rating)

    if rating >= 8:
        return "🌟 Excellent"

    elif rating >= 7:
        return "🔥 Very Good"

    elif rating >= 6:
        return "👍 Good"

    elif rating >= 5:
        return "🙂 Average"

    else:
        return "👎 Below Average"


# =============================================================================
# Rating Color
# =============================================================================

def rating_color(rating):

    rating = float(rating)

    if rating >= 8:
        return "#16a34a"

    elif rating >= 7:
        return "#3b82f6"

    elif rating >= 6:
        return "#eab308"

    elif rating >= 5:
        return "#f97316"

    return "#ef4444"


# =============================================================================
# Confidence Level
# =============================================================================

def confidence_level(rating):

    rating = float(rating)

    if rating >= 8:
        return "Very High"

    elif rating >= 7:
        return "High"

    elif rating >= 6:
        return "Medium"

    return "Low"


# =============================================================================
# Year Validation
# =============================================================================

def validate_year(year):

    current = datetime.now().year

    year = int(year)

    if year < 1913:
        return 1913

    if year > current:
        return current

    return year


# =============================================================================
# Duration Validation
# =============================================================================

def validate_duration(duration):

    duration = int(duration)

    if duration < 30:
        return 30

    if duration > 300:
        return 300

    return duration


# =============================================================================
# Votes Validation
# =============================================================================

def validate_votes(votes):

    votes = int(votes)

    if votes < 0:
        return 0

    return votes


# =============================================================================
# Format Votes
# =============================================================================

def format_votes(votes):

    votes = int(votes)

    if votes >= 1_000_000:
        return f"{votes/1_000_000:.1f}M"

    if votes >= 1000:
        return f"{votes/1000:.1f}K"

    return str(votes)


# =============================================================================
# Genre Count
# =============================================================================

def genre_count(genre):

    return len(str(genre).split(","))


# =============================================================================
# Multi Genre
# =============================================================================

def is_multigenre(genre):

    return genre_count(genre) > 1


# =============================================================================
# Movie Age
# =============================================================================

def movie_age(year):

    return datetime.now().year - int(year)


# =============================================================================
# Welcome Message
# =============================================================================

def welcome_message():

    hour = datetime.now().hour

    if hour < 12:
        return "☀️ Good Morning"

    elif hour < 17:
        return "🌤️ Good Afternoon"

    elif hour < 21:
        return "🌇 Good Evening"

    return "🌙 Good Night"
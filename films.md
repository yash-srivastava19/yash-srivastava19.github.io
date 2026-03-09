---
layout: page
title: Films
permalink: /films/
---

<link rel="stylesheet" href="/assets/css/films.css">

{% assign films = site.data.letterboxd.films %}

<div class="films-header">
  <span class="films-meta">Last synced: {{ site.data.letterboxd.last_updated | date: "%b %d, %Y" }}</span>
  <a class="films-lb-link" href="{{ site.letterboxd.profile_url }}" target="_blank" rel="noopener">Letterboxd ↗</a>
</div>

{% if films.size > 0 %}
{% assign rated = films | where_exp: "f", "f.rating_numeric != nil" %}
{% assign rewatch_count = films | where: "rewatch", true | size %}
{% assign total_rating = 0 %}
{% for f in rated %}{% assign total_rating = total_rating | plus: f.rating_numeric %}{% endfor %}
{% assign avg_rating = total_rating | divided_by: rated.size | round: 1 %}

<div class="films-stats">
  <div class="films-stat">
    <span class="films-stat-value">{{ films.size }}</span>
    <span class="films-stat-label">Watched</span>
  </div>
  <div class="films-stat">
    <span class="films-stat-value">{{ avg_rating }}</span>
    <span class="films-stat-label">Avg rating</span>
  </div>
  <div class="films-stat">
    <span class="films-stat-value">{{ rewatch_count }}</span>
    <span class="films-stat-label">Rewatches</span>
  </div>
  <div class="films-stat">
    <span class="films-stat-value">{{ films[0].watched_date | date: "%b %Y" }}</span>
    <span class="films-stat-label">Last watched</span>
  </div>
</div>

<div class="films-grid">
  {% for film in films %}
  <a class="film-card" href="{{ film.link }}" target="_blank" rel="noopener">
    {% if film.poster and film.poster != "" %}
    <img class="film-poster" src="{{ film.poster }}" alt="{{ film.title }}" loading="lazy">
    {% else %}
    <div class="no-poster">{{ film.title | slice: 0 }}</div>
    {% endif %}
    {% if film.rating_stars and film.rating_stars != "" %}
    <div class="film-badge">{{ film.rating_stars }}</div>
    {% endif %}
    <div class="film-overlay">
      <div class="film-rating">{{ film.rating_stars }}</div>
      <div class="film-title">{{ film.title }} <span class="film-year">{{ film.year }}</span></div>
      <div class="film-date">{{ film.watched_date }}{% if film.rewatch %}<span class="film-rewatch">rewatch</span>{% endif %}</div>
      {% if film.review and film.review != "" %}
      <div class="film-review">{{ film.review }}</div>
      {% endif %}
    </div>
  </a>
  {% endfor %}
</div>
{% else %}
<p>No films yet — check back after the first sync.</p>
{% endif %}

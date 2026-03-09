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

I watch a lot of films — mostly late at night, mostly alone, mostly with no plan. It started as something to do and became something I genuinely look forward to. I'm not a cinephile in the snobbish sense. I don't have a Letterboxd essay for every film. But I have opinions, and they're honest. Some of these have a review, most don't. The rating is gut feeling, not analysis. Hover over any poster to see what I thought.

{% if films.size > 0 %}
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

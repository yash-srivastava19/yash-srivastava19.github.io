Note: I wrote this in March 2026. A lot of developer friends around me were building constantly, staying sharp, committing code daily, and still could not get a job. It kept nagging at me. Why is the most committed group of builders somehow the most locked out? This is my attempt at making sense of that.

## The Unemployed Developer

There is something quietly strange happening in the developer world right now, and I don't think enough people are being honest about it.

The developers I know who are **unemployed** are not the lazy ones. They are not the ones who stopped learning, stopped building, or decided the market wasn't worth fighting for. Some of them have the greenest GitHub contribution graphs I have ever seen. Side projects, open-source contributions, tools built to scratch their own itch, commits at midnight because they genuinely could not stop. And still, month after month, the job market looks right through them.

Meanwhile, the companies that froze hiring are wrapping their remaining teams in AI tools, publishing quarterly earnings that beat expectations, and quoting CEOs who say AI will replace mid-level engineers "within 12-18 months." The message to the developer on the outside is: keep learning, keep building, keep proving yourself, but also maybe accept that the thing you are proving yourself at is being automated away.

This blog is about that paradox. The greenest commit graph with no job offer. The most curious builders being the most locked out of building professionally. And underneath all of it, what it actually reveals about what kind of developer survives this, and what kind of developer is just early.

---

## What The Numbers Actually Say

Let's start with the data, because the data is stark and the narrative around it has been fuzzy.

Software engineering job postings on Indeed fell **71% between February 2022 and August 2025**. That number deserves to sit on its own for a moment. Not a correction, not a rebalancing after a hiring surge. A 71% collapse in the signal that the market is looking for developers at all.

The collapse is not uniform. It hits hardest at the entry level, and the gap between junior and senior is not an accident.

<div style="margin: 32px 0; font-family: monospace;">
  <div style="margin-bottom: 8px; font-weight: bold; font-size: 0.95em;">Job Opportunity Decline (2022 to 2025)</div>
  <div style="margin-bottom: 10px; font-size: 0.8em; color: #888;">Sources: Indeed Hiring Lab, layoffs.fyi, Crunchbase, Stanford Digital Economy Lab</div>

  <div style="margin-bottom: 12px;">
    <div style="font-size: 0.85em; margin-bottom: 4px;">Software Dev Postings (Indeed)</div>
    <div style="display: flex; align-items: center; gap: 10px;">
      <div style="background: #c0392b; height: 22px; width: 71%; border-radius: 3px;"></div>
      <span style="font-size: 0.85em; font-weight: bold;">-71%</span>
    </div>
  </div>

  <div style="margin-bottom: 12px;">
    <div style="font-size: 0.85em; margin-bottom: 4px;">Entry-Level Jobs at Big Tech</div>
    <div style="display: flex; align-items: center; gap: 10px;">
      <div style="background: #e74c3c; height: 22px; width: 50%; border-radius: 3px;"></div>
      <span style="font-size: 0.85em; font-weight: bold;">-50%</span>
    </div>
  </div>

  <div style="margin-bottom: 12px;">
    <div style="font-size: 0.85em; margin-bottom: 4px;">Overall Tech Job Postings (all roles)</div>
    <div style="display: flex; align-items: center; gap: 10px;">
      <div style="background: #e67e22; height: 22px; width: 36%; border-radius: 3px;"></div>
      <span style="font-size: 0.85em; font-weight: bold;">-36%</span>
    </div>
  </div>

  <div style="margin-bottom: 12px;">
    <div style="font-size: 0.85em; margin-bottom: 4px;">Tech Internship Postings</div>
    <div style="display: flex; align-items: center; gap: 10px;">
      <div style="background: #e67e22; height: 22px; width: 30%; border-radius: 3px;"></div>
      <span style="font-size: 0.85em; font-weight: bold;">-30%</span>
    </div>
  </div>

  <div style="margin-bottom: 12px;">
    <div style="font-size: 0.85em; margin-bottom: 4px;">Senior/Manager Roles</div>
    <div style="display: flex; align-items: center; gap: 10px;">
      <div style="background: #f39c12; height: 22px; width: 19%; border-radius: 3px;"></div>
      <span style="font-size: 0.85em; font-weight: bold;">-19%</span>
    </div>
  </div>
</div>

The ADP Research Institute tracked 75,000+ software developers across 6,500 companies from January 2018 to January 2024. By January 2024, U.S. software developer employment was **17% below its January 2018 level**. Not below a pandemic peak. Below a 2018 baseline. The industry was employing fewer developers than it did six years prior, at the peak of a tech boom that produced record stock prices and record AI investment.

A Stanford study published in August 2025 - running on actual payroll records from millions of workers, not survey data - found a **13% relative employment decline for early-career workers in AI-exposed jobs** since generative AI went mainstream. For software developers specifically, employment for workers aged 22-25 dropped **nearly 20% from its 2022 peak**. Older, experienced developers in the same occupations saw **6-9% employment growth** in the same period.

The market didn't kill developers. It killed the pipeline.

<div style="margin: 32px 0; font-family: monospace;">
  <div style="margin-bottom: 8px; font-weight: bold; font-size: 0.95em;">Tech Layoffs Citing AI as Direct Cause</div>
  <div style="margin-bottom: 10px; font-size: 0.8em; color: #888;">Source: Challenger, Gray & Christmas</div>

  <div style="margin-bottom: 12px;">
    <div style="font-size: 0.85em; margin-bottom: 4px;">2023</div>
    <div style="display: flex; align-items: center; gap: 10px;">
      <div style="background: #95a5a6; height: 22px; width: 4%; border-radius: 3px; min-width: 6px;"></div>
      <span style="font-size: 0.85em;">~4,600 layoffs</span>
    </div>
  </div>

  <div style="margin-bottom: 12px;">
    <div style="font-size: 0.85em; margin-bottom: 4px;">2024</div>
    <div style="display: flex; align-items: center; gap: 10px;">
      <div style="background: #e67e22; height: 22px; width: 22%; border-radius: 3px;"></div>
      <span style="font-size: 0.85em;">~12,000 layoffs</span>
    </div>
  </div>

  <div style="margin-bottom: 12px;">
    <div style="font-size: 0.85em; margin-bottom: 4px;">2025</div>
    <div style="display: flex; align-items: center; gap: 10px;">
      <div style="background: #c0392b; height: 22px; width: 100%; border-radius: 3px;"></div>
      <span style="font-size: 0.85em; font-weight: bold;">54,836 layoffs — 12x growth in two years</span>
    </div>
  </div>
</div>

Over 244,000 people lost tech jobs in 2025. AI was the explicitly stated reason for 54,836 of those. Two years prior, that number was around 4,600. The number isn't just growing. It is accelerating.

---

## The Green Graph Nobody Sees

Here is the cruelest irony inside all of this data.

If you look at the GitHub contribution graph of someone who is currently employed at a tech company working in a private monorepo, you will often see a nearly empty public profile. Their best work, the production code they are shipping daily, lives in a private repository. The graph tells you nothing about them.

Now look at the developer who has been unemployed for six months. They have been building side projects to stay sharp, contributing to open source to show they are still shipping, committing daily because they do not know what else to do. Their graph is solid green. Every square lit up. A record of anxious productivity.

The hiring system built a signal that rewards the wrong thing. The greenest public graphs often belong to the most anxious people, developers who have been locked out of the market and are performing productivity in public because the system trained them to believe that is what gets noticed.

Reports from developer communities in 2024 and 2025 told the same stories on a loop: 500 applications, zero callbacks. 150 companies, one recruiter screen. Offers rescinded mid-process because of hiring freezes. Computer science graduates from strong programs applying to hundreds of roles and hearing nothing for months. CNN ran a feature in August 2025 titled "150 job applications, rescinded offers: Computer science grads are struggling to find work." It was not an isolated story. It was the modal experience for an entire graduating class.

The contribution graph is not the only gamed metric. LeetCode ratings, hackathon badges, certification farms - all of it became a substitute for the thing companies say they want, which is developers who can actually reason about hard problems. When you optimize hard enough for a signal, the signal decouples from what it was supposed to measure. The system ends up selecting for the performance of skill, not skill itself.

And the deepest irony: the ones grinding hardest on these signals are the ones who cannot afford to stop. The employed developer doesn't need to. The unemployed one can't stop.

---

## What The CEOs Said

It would be unfair not to quote the people who have been most vocal about what is coming.

Dario Amodei, CEO of Anthropic, in late 2024:

> "We might be 6-12 months away from models doing all of what software engineers do end-to-end."

He also predicted AI would write **90% of code by 2025** and that AI could eliminate half of all entry-level white-collar jobs within five years.

Mark Zuckerberg, on the Joe Rogan Experience:

> "Probably in 2025, we at Meta are going to have an AI that can effectively be a sort of mid-level engineer at your company that can write code."

Jensen Huang at the World Government Summit:

> "With AI, no one will require Java or C to do coding."

He advised younger generations not to learn to code at all, and suggested biology, education, or farming as more durable paths.

Sam Altman described the coming generation of AI agents as "junior employees" - entities you assign tasks to, review the output of, and send back with feedback.

These are not random commentators. These are the people building the tools that are changing the market. Their predictions deserve engagement, not dismissal. But they also are not the full picture, and taking them at face value has created something toxic in developer culture: the belief that skills are being deprecated in real time, and that the only rational response is to fold yourself entirely into AI workflows or accept being replaced.

The Klarna story is a useful case study. Klarna went from 5,527 employees to 3,422 over two years, a 40% workforce reduction, with AI cited as a primary driver. Their AI customer service bot handled the equivalent of 700 full-time agents' work in its first month. CEO Sebastian Siemiatkowski was the most candid voice in the industry, saying other tech CEOs were "sugarcoating" the displacement. He called it a coming recession in white-collar work.

Then Klarna had to rehire. Product quality had degraded. Service had gotten measurably worse. The efficiency gains had eaten something that wasn't visible in a spreadsheet: institutional knowledge, judgment on edge cases, the human threads that hold together a customer relationship over time.

That reversal is not a footnote. It is a real data point about what gets lost when you reduce human work to its most automatable surface.

---

## The Productivity Numbers Don't Add Up

The sales pitch for AI coding tools is roughly: **10x productivity, no downside.** The empirical research says something far more complicated.

METR ran a randomized controlled trial in July 2025, designed explicitly to measure whether the most capable AI tools genuinely help experienced developers. Sixteen experienced open-source developers. 246 real tasks on their own codebases. Best available tools: Cursor Pro with Claude 3.5/3.7 Sonnet. The result was not what anyone expected.

**AI made these developers 19% slower, not faster.** Not on average, not for some subset of tasks. Slower, measured objectively. The developers themselves *believed* they were working faster. The measured task completion times went up.

GitClear spent four years analyzing 211 million lines of code (2020-2024) and found:

| Metric | 2020 | 2024 | Change |
|--------|------|------|--------|
| Code churn rate (revised within 2 weeks) | 3.1% | 5.7% | +84% |
| Refactored/restructured code | 24.1% | 9.5% | -61% |
| Copy-pasted code block frequency | baseline | 8x | +700% |

The code isn't better. It is more voluminous. It is churning faster. It is being copy-pasted at 8x the historical rate. The architecture is getting shallower.

Stack Overflow's 2025 survey of 49,000 developers found:
- **66%** cite "AI solutions that are almost right, but not quite" as their biggest time sink
- Developer trust in AI accuracy is at an **all-time low of 29%**
- **45%** say debugging AI-generated code is more time-consuming than it saves

GitHub says Copilot is now generating **46% of all code in Copilot-enabled projects**. Microsoft says 20-30% of their entire codebase is AI-written. Google says the same. And Cursor hit **$2 billion ARR in February 2026**, with over a million daily active users.

So the picture is: companies are heavily adopting AI coding tools, cutting developer headcount, and producing code with higher churn rates, lower architectural quality, and significantly more copy-paste. The developers left standing are spending significant chunks of their time debugging AI output, a task that is less satisfying and slower than writing clean code in the first place.

This is not the productivity revolution it was sold as. Not yet. It might get there. Right now, the biggest measurable effect of AI on software teams is that they have fewer people, more code, and less structural integrity in the codebase.

---

## The Thing About Caring About The Problem

Here is the part I feel most strongly about, and want to be honest rather than optimistic about.

The developers I have watched actually thrive in this period - not just survive, but do their best and most interesting work - are not the ones who optimized their job hunt. They are not the ones with the most updated LinkedIn profiles or the highest LeetCode contest ratings. They are the ones who found a problem they genuinely cared about and went unreasonably deep on it.

This is not a motivational observation. It is an observation about what the loop looks like when it works versus when it does not.

When you care about a problem, you build something real. When you build something real, you hit the hard parts: the edge cases, the architectural decisions, the moments where the naive solution breaks and you have to actually think. When you actually think, you develop judgment. Judgment is the thing that does not compress into a prompt, does not get generated by Copilot, and does not show up on a contribution graph.

**Andrej Karpathy** didn't build nanoGPT and micrograd as job interview prep. He built them because he wanted to understand something deeply, and he wanted to share that understanding with other people. nanoGPT has been forked hundreds of thousands of times. It is one of the primary reasons an entire generation of developers has a real mental model of how transformers work, not a surface-level understanding of how to call the API. That's the loop working correctly: care about the problem, go deep, build something honest, the work finds its audience.

**The four-person team at Anysphere** built Cursor. They didn't set out to put a suggestion overlay on VS Code. They had a specific, genuine belief about what an AI-native editor should feel like: not autocomplete on top of an existing tool, but something that actually understands the codebase as a whole. They went deep on that specific belief. Cursor went from a $2.5 billion valuation to roughly $30 billion in a single year, not because they were first - GitHub Copilot was years ahead of them - but because they had more conviction about the right problem.

**Simon Willison** has been one of the most honest and productive voices in the AI era. He ships tools constantly, writes about them honestly, is transparent about what works and what does not, and has built a body of work that compounds across years. The key is not that he knows more tools than anyone else. It is that he approaches every problem with genuine curiosity and publishes what he finds without trying to fit it into a narrative.

The common thread across all of these is not AI fluency. It is not staying current with the latest model releases. It is that the work is driven by a genuine problem that will not let go of the builder. That loop, care about the problem deeply enough to build honestly, is the thing the AI market cannot replicate and the hiring market is genuinely bad at measuring.

When you are in that loop, the AI tools are genuinely useful. Not because they make you 10x faster, but because they remove friction on the parts you already understand, giving you more time on the parts you do not. The developer who understands the problem is the one who can tell the AI exactly what to build and know immediately when the output is wrong.

---

## On Top of AI, Not Under It

The developers being most displaced right now are the ones doing the task AI is genuinely best at: generating average solutions to well-specified problems. Junior developers assigned to write boilerplate, fill CRUD endpoints, translate requirements into code that looks like every other codebase - yes, that work compresses well. That is not a conspiracy. That is what the tools are optimized for.

The developers who are not replaceable are the ones who understand the problem deeply enough to know when the AI is wrong. Who can look at 500 lines of generated code and identify the 40 that will cause a production incident. Who can architect something that has not been built before, because the problem itself is new.

The practical principle I keep coming back to: **use AI aggressively, but never lose the thread back to first principles.** Build things from scratch sometimes, not because it is efficient, but because it keeps the mental model intact. The developers who have actually built a version control system understand Git in a way that no amount of muscle-memory `git rebase` practice can replicate. The developers who have implemented backpropagation by hand understand transformer behavior in a way that no amount of prompt engineering substitutes for.

The risk is not that AI writes code and you do not. The risk is that you stop understanding what you are building. That you become a reviewer of AI output without ever developing the judgment to know when it is confidently wrong, and AI is confidently wrong more often than the 29% trust number implies, because it fails on the exact problems where you most need it to be right.

Being on top of AI means: you set the direction, you understand the output, you catch the errors that count, and you deliberately take on the problems that genuinely require human judgment. That category is shrinking. It is not empty. And the developers who have been working through it with real curiosity are building something the market has not yet learned to price correctly.

<div style="margin: 32px 0; font-family: monospace; font-size: 0.9em;">
  <div style="font-weight: bold; margin-bottom: 12px;">The Loop: What It Looks Like From Both Sides</div>
  <div style="display: flex; gap: 24px; flex-wrap: wrap;">
    <div style="flex: 1; min-width: 220px; border-left: 3px solid #c0392b; padding-left: 12px;">
      <div style="font-weight: bold; margin-bottom: 8px; color: #c0392b;">The Signal Game</div>
      <div style="line-height: 1.8;">Green commit graph for visibility<br>LeetCode grind for interviews<br>AI tools to look productive<br>Certifications for checkboxes<br>Performance of skill, not skill</div>
      <div style="margin-top: 10px; font-style: italic; color: #888;">Outcome: Treadmill. Goalposts move.</div>
    </div>
    <div style="flex: 1; min-width: 220px; border-left: 3px solid #27ae60; padding-left: 12px;">
      <div style="font-weight: bold; margin-bottom: 8px; color: #27ae60;">The Craft Loop</div>
      <div style="line-height: 1.8;">Care about a real problem<br>Build something honest<br>Hit the hard parts, develop judgment<br>AI augments what you already understand<br>Work compounds over time</div>
      <div style="margin-top: 10px; font-style: italic; color: #888;">Outcome: Something that actually matters.</div>
    </div>
  </div>
</div>

---

## Why Any Of This Matters

I did not write this to be bleak. The situation is genuinely hard for a lot of developers right now and I am not going to pretend otherwise. The entry-level collapse is real. The hiring freeze is real. The companies that said AI would let them need fewer people, then found out service quality degraded, are real, and their lesson takes time to propagate through the rest of the industry.

But the part that matters more than any of the job market data is what kind of developer you are building yourself into during this period.

The developers treating this as a crisis of credentials are going to keep running on a treadmill that does not stop. More tools, more signals, more things to learn in the hope that the bar stops moving. It will not. The bar moves because the thing being measured is not the thing that actually matters.

The developers treating this as an opportunity to go deep on problems they genuinely care about are building something that does not depend on the hiring market to validate it. The work compounds. The judgment accumulates. The loop, once you are in it, becomes self-sustaining in a way that no amount of signal optimization can replicate.

The greenest commit graph should not belong to the most anxious developer. It should belong to the most curious one.

There is a developer somewhere right now building something unreasonably specific. Something that solves a problem they found while going deep on something they cared about. Something where the architecture required actually thinking, where the AI tools helped but couldn't do the hard part, where the result is honest and strange and exactly what it needed to be.

That developer is not unemployable. That developer is just early.

---

*Data sources: Stanford Digital Economy Lab (Brynjolfsson, Chan, Chen, Aug 2025), METR Randomized Controlled Trial (Jul 2025), GitClear Code Quality Analysis (2024), Stack Overflow Developer Survey 2025, GitHub Copilot Stats (Jul 2025), Challenger Gray & Christmas (2025 Year-End Report), Indeed Hiring Lab (Jul 2025), ADP Research Institute (2024), Crunchbase/layoffs.fyi.*

"""The production overseer: watch deployed traffic cheaply and send failures back to the lab.

Traces arrive through the Supabase buffer (pulled here), the local ingest route or a JSONL
upload. Tier one runs free deterministic checks on every trace. Tier two asks a small judge
model about suspicious traces plus a fixed random sample, under a hard daily budget. Reviewers
label what matters, and labeled failures become an ordinary production scan, so the existing
train-on-half, measure-the-other-half loop applies unchanged.
"""

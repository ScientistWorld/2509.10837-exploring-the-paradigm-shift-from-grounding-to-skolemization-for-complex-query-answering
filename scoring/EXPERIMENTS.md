# Experiment Overview

## scaled_synthetic_cqa
Tests whether the reproduced system can answer non-trivial complex queries over
a fixed incomplete knowledge graph. The experiment includes atomic lookup,
multi-hop paths, intersections, and negated intersections so that success
requires more than one-hop memorization. It also reports model size to preserve
the accuracy-with-resource-control goal. The same experiment can include
method variants to test whether design choices affect negated-query behavior.
When prediction files include optional trace fields, the same evaluator also
checks whether intermediate ranked entities are valid witnesses for multi-hop
reasoning chains.

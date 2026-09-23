# Shared body for the distance_*.test drivers.
#
# Adapted from the server's coverage of the built-in DISTANCE(v1, v2, metric)
# function.  The SVECTOR extension has no metric argument: each metric is its
# own VDF, so the driver sets $metric and this file maps it to $func.
#
#   EUCLIDEAN -> L2_DISTANCE      MANHATTAN -> L1_DISTANCE
#   COSINE    -> COSINE_DISTANCE  DOT       -> INNER_PRODUCT
#
# EUCLIDEAN_SQUARED has no SQL entry point: svector_distance_l2_squared is
# registered only as an HNSW index helper, so there is no driver for it.
#
# Other differences from the server's DISTANCE() that shape this file:
#   - Values are built with SVECTOR::FROM_STRING(), not TO_VECTOR().  A bare
#     string constant also works as long as the other argument pins the
#     dimension; binary strings (X'...') are not accepted at all.
#   - SVECTOR columns are strictly dimensioned, so there is no runtime
#     length-mismatch case: a mismatch is either rejected when the row is
#     inserted or when the function is resolved.
#   - All function-resolution failures surface as ER_VILLAGESQL_GENERIC_ERROR.
#   - NaN/Inf inputs do not raise ER_DATA_OUT_OF_RANGE; they propagate.
#
# Distance values use integer-valued components whose partial sums stay well
# under 2^24, so every float32 add in the SIMD kernels is exact and the
# recorded results are identical on every dispatch tier.  See AGENTS.md.

if ($metric == EUCLIDEAN)
{
  let $func = L2_DISTANCE;
}
if ($metric == MANHATTAN)
{
  let $func = L1_DISTANCE;
}
if ($metric == COSINE)
{
  let $func = COSINE_DISTANCE;
}
if ($metric == DOT)
{
  let $func = INNER_PRODUCT;
}

--echo #
--echo # Test coverage for SVECTOR $func().
--echo #

SET PERSIST vsql_allow_preview_extensions = ON;
INSTALL EXTENSION vsql_vector;

--echo #
--echo # 0) Prepare playground.
--echo #
CREATE TABLE t1 (id INT PRIMARY KEY, v1 SVECTOR(1), v2 SVECTOR(2)) ENGINE=InnoDB;
INSERT INTO t1 VALUES (0, '[0]', '[0, 0]'),
                      (1, '[1]', '[1, 0]'),
                      (2, '[1]', '[0, 1]'),
                      (3, '[2]', '[1, 1]'),
                      (4, '[2]', '[2, 0]'),
                      (99, NULL, NULL);
--echo # SVECTOR columns are strictly dimensioned: a shorter value is rejected on
--echo # insert, so unlike VECTOR there is no stored value whose length differs
--echo # from the column definition.
--error ER_TRUNCATED_WRONG_VALUE_FOR_FIELD
INSERT INTO t1 VALUES (98, '[1]', '[2]');

--echo #
--echo # 1) Test how different number and types of arguments are handled.
--echo #
--echo # 1.1) Arity. The VDF takes exactly two arguments; there is no metric
--echo #      argument to get wrong.
--echo #
--error ER_VILLAGESQL_GENERIC_ERROR
eval SELECT $func();
--error ER_VILLAGESQL_GENERIC_ERROR
eval SELECT $func(SVECTOR::FROM_STRING('[1]'));
eval SELECT $func(SVECTOR::FROM_STRING('[1]'), SVECTOR::FROM_STRING('[2]'));
--error ER_VILLAGESQL_GENERIC_ERROR
eval SELECT $func(SVECTOR::FROM_STRING('[1]'), SVECTOR::FROM_STRING('[2]'), 'EUCLIDEAN');

--echo #
--echo # 1.2) Argument types.
--echo #
--echo # A string constant is accepted when the other argument pins the
--echo # dimension, but two bare constants leave it undetermined.
eval SELECT $func('[1]', SVECTOR::FROM_STRING('[2]'));
eval SELECT $func(SVECTOR::FROM_STRING('[1]'), '[2]');
eval SELECT $func('[0, 0]', v2) FROM t1 WHERE id = 4;
eval SELECT $func(v2, '[0, 0]') FROM t1 WHERE id = 4;
--error ER_VILLAGESQL_GENERIC_ERROR
eval SELECT $func('[1]', '[2]');
--echo # Anything that is neither an SVECTOR nor a string constant is rejected.
--error ER_VILLAGESQL_GENERIC_ERROR
eval SELECT $func(id, SVECTOR::FROM_STRING('[2]')) FROM t1 WHERE id = 0;
--error ER_VILLAGESQL_GENERIC_ERROR
eval SELECT $func(SVECTOR::FROM_STRING('[0]'), id) FROM t1 WHERE id = 1;
--echo # Binary strings are not a valid SVECTOR source (unlike VECTOR).
--error ER_VILLAGESQL_GENERIC_ERROR
eval SELECT $func(X'0000803F', SVECTOR::FROM_STRING('[2]'));

--echo #
--echo # 1.3) NULL arguments and nullability in metadata for result.
--echo #
eval SELECT $func(NULL, SVECTOR::FROM_STRING('[1, 0]'));
eval SELECT $func(SVECTOR::FROM_STRING('[1, 1]'), NULL);
eval SELECT $func(v2, SVECTOR::FROM_STRING('[1, 0]')) FROM t1 WHERE id = 99;
eval SELECT $func(SVECTOR::FROM_STRING('[1, 1]'), v2) FROM t1 WHERE id = 99;
--echo # The result metadata should indicate that it is nullable.
eval CREATE TABLE tt SELECT $func(SVECTOR::FROM_STRING('[1, 0]'),
                                  SVECTOR::FROM_STRING('[0, 1]')) AS d;
SHOW CREATE TABLE tt;
DROP TABLE tt;

--echo #
--echo # 2) Test vector arguments dimension mismatch.
--echo #
--echo # Unlike VECTOR, the mismatch is caught while resolving the function --
--echo # the dimension is part of the type -- so it never reaches the kernel.
--error ER_VILLAGESQL_GENERIC_ERROR
eval SELECT $func(SVECTOR::FROM_STRING('[1]'), SVECTOR::FROM_STRING('[1, 0]'));
--error ER_VILLAGESQL_GENERIC_ERROR
eval SELECT $func(v2, SVECTOR::FROM_STRING('[1]')) FROM t1 WHERE id = 1;
--error ER_VILLAGESQL_GENERIC_ERROR
eval SELECT $func(v1, v2) FROM t1 WHERE id = 1;
--echo # A string constant inherits the dimension of the other argument, so a
--echo # constant of the wrong length fails to convert instead -- a value error
--echo # rather than a type-parameter conflict.
--error ER_WRONG_VALUE
eval SELECT $func(v2, '[1]') FROM t1 WHERE id = 1;

--echo #
--echo # 3) Some basic tests for different (from syntax PoV) variants of
--echo #    arguments.
--echo #
eval SELECT $func(SVECTOR::FROM_STRING('[0, 0]'), SVECTOR::FROM_STRING('[1, 0]'));
eval SELECT $func(SVECTOR::FROM_STRING('[0, 0]'), v2) FROM t1 WHERE id = 1;
eval SELECT $func(a.v2, b.v2) FROM t1 AS a, t1 AS b WHERE a.id = 0 AND b.id = 4;
eval SELECT $func(v2, SVECTOR::FROM_STRING('[0, 2]')) FROM t1 WHERE id = 0;
--echo # Non-trivial (artificial) combinations.
eval SELECT $func(SVECTOR::FROM_STRING(CONCAT('[0', ', ', '1]')),
                  SVECTOR::FROM_STRING('[0, 2]'));

--echo #
--echo # 4) Basic test for different vector values.
--echo #
--echo # Identical / collinear vectors.
eval SELECT $func(SVECTOR::FROM_STRING('[1, 1]'), SVECTOR::FROM_STRING('[1, 1]'));
eval SELECT $func(SVECTOR::FROM_STRING('[1, 0]'), SVECTOR::FROM_STRING('[1, 0]'));
eval SELECT $func(SVECTOR::FROM_STRING('[1, 1]'), SVECTOR::FROM_STRING('[2.5, 2.5]'));
eval SELECT $func(SVECTOR::FROM_STRING('[1, 2, 3, 4, 5]'),
                  SVECTOR::FROM_STRING('[1, 2, 3, 4, 5]'));
--echo # Orthogonal vectors.
eval SELECT $func(SVECTOR::FROM_STRING('[1, 0]'), SVECTOR::FROM_STRING('[0, 1]'));
eval SELECT $func(SVECTOR::FROM_STRING('[0, 1, 0]'), SVECTOR::FROM_STRING('[-1, 0, -1]'));
eval SELECT $func(SVECTOR::FROM_STRING('[1, 0, 3, 0, 5]'),
                  SVECTOR::FROM_STRING('[0, 2, 0, 4, 0]'));
--echo # Anti-parallel vectors.
eval SELECT $func(SVECTOR::FROM_STRING('[-1, -1]'), SVECTOR::FROM_STRING('[2, 2]'));
--echo # Distance from origin.
eval SELECT $func(SVECTOR::FROM_STRING('[0, 0]'), SVECTOR::FROM_STRING('[1, 0]'));
eval SELECT $func(SVECTOR::FROM_STRING('[0, 0]'), SVECTOR::FROM_STRING('[3, 4]'));
eval SELECT $func(SVECTOR::FROM_STRING('[0, 0]'), SVECTOR::FROM_STRING('[5, 12]'));
eval SELECT $func(SVECTOR::FROM_STRING('[0, 0, 0, 0]'), SVECTOR::FROM_STRING('[1, 1, 1, 1]'));
--echo # Mixed-sign and larger vectors.
eval SELECT $func(SVECTOR::FROM_STRING('[1, 2, 3]'), SVECTOR::FROM_STRING('[4, 5, 6]'));
eval SELECT $func(SVECTOR::FROM_STRING('[1, 7, 3, 16, 5]'),
                  SVECTOR::FROM_STRING('[1, 2, 3, 4, 5]'));
--echo # Zero vector (behavior differs per metric; cosine is undefined and
--echo # returns 1, its maximum distance, rather than NULL).
eval SELECT $func(SVECTOR::FROM_STRING('[0, 0]'), SVECTOR::FROM_STRING('[2, 2]'));
eval SELECT $func(SVECTOR::FROM_STRING('[0, 0]'), SVECTOR::FROM_STRING('[0, 0]'));
eval SELECT $func(SVECTOR::FROM_STRING('[0]'), SVECTOR::FROM_STRING('[0]'));
--echo # Symmetry: f(a, b) = f(b, a).
eval SELECT $func(SVECTOR::FROM_STRING('[1, 2, 3]'), SVECTOR::FROM_STRING('[4, 5, 6]')) =
            $func(SVECTOR::FROM_STRING('[4, 5, 6]'), SVECTOR::FROM_STRING('[1, 2, 3]'))
       AS symmetric;

--echo #
--echo # 5) Wide-tier SIMD coverage (dims >= 16 dispatches to the wide kernel).
--echo #
--echo # 16-dim: fills one AVX-512 register / two AVX2 / four SSE4.2 -- no tail.
eval SELECT $func(SVECTOR::FROM_STRING('[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]'),
                  SVECTOR::FROM_STRING('[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]'));
--echo # 20-dim: SSE4.2 5x4 (no tail); AVX2 2x8 + 4-elem tail;
--echo #         AVX-512 1x16 + 4-elem tail.
eval SELECT $func(SVECTOR::FROM_STRING('[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]'),
                  SVECTOR::FROM_STRING('[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]'));

--echo #
--echo # 6) Special float32 values.
--echo #
--echo # Elements outside the float32 range are rejected while parsing.
--error ER_VILLAGESQL_GENERIC_ERROR
SELECT SVECTOR::FROM_STRING('[1e40]');
--echo # NaN and the infinities, by contrast, are accepted and stored verbatim.
SELECT VECTOR_HEX(SVECTOR::FROM_STRING('[nan]')) AS nan_hex,
       VECTOR_HEX(SVECTOR::FROM_STRING('[inf]')) AS inf_hex,
       VECTOR_HEX(SVECTOR::FROM_STRING('[-inf]')) AS neg_inf_hex;
--echo # They propagate into the result instead of raising an error.  Printing
--echo # cannot show that: the client protocol renders NaN and +-Inf alike as 0,
--echo # and a bound test alone does not separate them either, because MySQL's
--echo # real comparator ranks NaN above every finite value.  Classify instead
--echo # by self-equality plus a bound, reading the value out of a derived table
--echo # -- a NaN does not survive materialisation and arrives as NULL, which is
--echo # itself the signal:
--echo #   (NULL, NULL) NaN     (1, 1) +Inf     (1, 0) finite
eval SELECT d = d AS self_equal, d >= 1e300 AS at_or_above_bound
     FROM (SELECT $func(SVECTOR::FROM_STRING('[nan]'),
                        SVECTOR::FROM_STRING('[0]')) AS d) AS nan_case;
eval SELECT d = d AS self_equal, d >= 1e300 AS at_or_above_bound
     FROM (SELECT $func(SVECTOR::FROM_STRING('[inf]'),
                        SVECTOR::FROM_STRING('[0]')) AS d) AS inf_case;

--echo #
--echo # 7) Values near the float32 maximum.
--echo #
--echo # Squaring 2e38 overflows float32, but the kernels detect the non-finite
--echo # reduction and recompute in double, so the result stays finite.
eval SELECT $func(SVECTOR::FROM_STRING('[0, 0]'), SVECTOR::FROM_STRING('[2e38, 0]'));
--echo # Same value in a 16-dim vector, i.e. through the wide kernel.
eval SELECT $func(SVECTOR::FROM_STRING('[2e38,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'),
                  SVECTOR::FROM_STRING('[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'));
--echo # Anti-parallel pair whose element-wise difference (4e38) itself overflows
--echo # float32.  Even the double-precision fallback forms each difference in
--echo # float32 first (euclidean_scalar: "const double d = av - bv"), so
--echo # L2_DISTANCE saturates to +Inf here.  L1_DISTANCE casts before
--echo # subtracting and INNER_PRODUCT multiplies rather than subtracts, so both
--echo # stay finite; COSINE_DISTANCE never differences its inputs at all.
--echo # Same (self_equal, at_or_above_bound) encoding as section 6.
eval SELECT d = d AS self_equal, d >= 1e300 AS at_or_above_bound
     FROM (SELECT $func(SVECTOR::FROM_STRING('[-2e38, 1]'),
                        SVECTOR::FROM_STRING('[2e38, -1]')) AS d) AS overflow_case;

--echo #
--echo # 8) Prepared statements.
--echo #
eval PREPARE s1 FROM 'SELECT $func(v1, SVECTOR::FROM_STRING("[2]")) FROM t1 WHERE id = 4';
EXECUTE s1;
EXECUTE s1;
DEALLOCATE PREPARE s1;
eval PREPARE s2 FROM 'SELECT $func(v1, SVECTOR::FROM_STRING(?)) FROM t1 WHERE id = 4';
SET @vec = '[2]';
EXECUTE s2 USING @vec;
DEALLOCATE PREPARE s2;

--echo #
--echo # 9) Distance in query contexts.
--echo #
--echo # ORDER BY distance: nearest-neighbour pattern.
eval SELECT id FROM t1 WHERE id IN (0,1,2,3,4)
     ORDER BY $func(v2, SVECTOR::FROM_STRING('[1, 0]')), id;
--echo # ORDER BY distance DESC: farthest-neighbour pattern.
eval SELECT id FROM t1 WHERE id IN (0,1,2,3,4)
     ORDER BY $func(v2, SVECTOR::FROM_STRING('[1, 0]')) DESC, id;
--echo # WHERE: range query filtering by distance.
eval SELECT id FROM t1
     WHERE id IN (0,1,2,3,4) AND $func(v2, SVECTOR::FROM_STRING('[1, 0]')) < 1.5
     ORDER BY id;
--echo # Derived table with distance.
eval SELECT id FROM
     (SELECT id, $func(v2, SVECTOR::FROM_STRING('[1, 0]')) AS d
      FROM t1 WHERE id IN (0,1,2,3,4)) AS sq
     WHERE d IS NOT NULL ORDER BY d, id;

if ($metric == COSINE)
{
--echo #
--echo # 10) Zero-vector cosine in DML: inserts 1, the documented maximum
--echo #     distance, without aborting under strict sql_mode.
--echo #
CREATE TABLE tt_cosine_dml (d DOUBLE) ENGINE=InnoDB;
INSERT INTO tt_cosine_dml
  SELECT COSINE_DISTANCE(SVECTOR::FROM_STRING('[0]'), SVECTOR::FROM_STRING('[0]'));
SELECT d FROM tt_cosine_dml;
DROP TABLE tt_cosine_dml;
}

DROP TABLE t1;

UNINSTALL EXTENSION vsql_vector;
SET PERSIST vsql_allow_preview_extensions = OFF;
RESET PERSIST vsql_allow_preview_extensions;

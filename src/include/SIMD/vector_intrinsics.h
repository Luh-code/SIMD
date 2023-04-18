#ifndef __SIMD_VECTOR_INTRINSICS_H__
#define __SIMD_VECTOR_INTRINSICS_H__

#include <emmintrin.h>
#include <stdint.h>

typedef union
{
  __m128 v;
  float n[4];
} simd_float4;

static inline void simd_set_float4(simd_float4& res, float a, float b, float c, float d)
{
  res.v = _mm_set_ps(a, b, c, d);
}

static inline void simd_set_float4(simd_float4& res, float v[4])
{
  res.v = _mm_set_ps(v[0], v[1], v[2], v[3]);
}

static inline void simd_set_float4(simd_float4& res, float a)
{
  res.v = _mm_set1_ps(a);
}

static inline void simd_add_float4(simd_float4& res, simd_float4& a, simd_float4& b)
{
  res.v = _mm_add_ps(a.v, b.v);
}

static inline void simd_add_float4(simd_float4& a, simd_float4& b)
{
  simd_add_float4(a, a, b);
}

static inline void simd_sub_float4(simd_float4& res, simd_float4& a, simd_float4& b)
{
  res.v = _mm_sub_ps(a.v, b.v);
}

static inline void simd_sub_float4(simd_float4& a, simd_float4& b)
{
  simd_sub_float4(a, a, b);
}

static inline void simd_mul_float4(simd_float4& res, simd_float4& a, simd_float4& b)
{
  res.v = _mm_mul_ps(a.v, b.v);
}

static inline void simd_mul_float4(simd_float4& a, simd_float4& b)
{
  simd_mul_float4(a, a, b);
}

static inline void simd_div_float4(simd_float4& res, simd_float4& a, simd_float4& b)
{
  res.v = _mm_div_ps(a.v, b.v);
}

static inline void simd_div_float4(simd_float4& a, simd_float4& b)
{
  simd_div_float4(a, a, b);
}

static inline void simd_sqrt_float4(simd_float4& res, simd_float4& a)
{
  res.v = _mm_sqrt_ps(a.v);
}

static inline void simd_sqrt_float4(simd_float4& a)
{
  simd_sqrt_float4(a, a);
}

static inline void simd_copy_float4(simd_float4& dst, simd_float4& a)
{
  _mm_storeu_ps(dst.n, a.v);
}

typedef union
{
  __m128d v;
  double n[2];
} simd_double2;

static inline void simd_set_double2(simd_double2& res, double a, double b)
{
  res.v = _mm_set_pd(a, b);
}

static inline void simd_set_double2(simd_double2& res, double v[2])
{
  res.v = _mm_set_pd(v[0], v[1]);
}

static inline void simd_set_double2(simd_double2& res, double a)
{
  res.v = _mm_set1_pd(a);
}

static inline void simd_add_double2(simd_double2& res, simd_double2& a, simd_double2& b)
{
  res.v = _mm_add_pd(a.v, b.v);
}

static inline void simd_add_double2(simd_double2& a, simd_double2& b)
{
  simd_add_double2(a, a, b);
}

static inline void simd_sub_double2(simd_double2& res, simd_double2& a, simd_double2& b)
{
  res.v = _mm_sub_pd(a.v, b.v);
}

static inline void simd_sub_double2(simd_double2& a, simd_double2& b)
{
  simd_sub_double2(a, a, b);
}

static inline void simd_mul_double2(simd_double2& res, simd_double2& a, simd_double2& b)
{
  res.v = _mm_mul_pd(a.v, b.v);
}

static inline void simd_mul_double2(simd_double2& a, simd_double2& b)
{
  simd_mul_double2(a, a, b);
}

static inline void simd_div_double2(simd_double2& res, simd_double2& a, simd_double2& b)
{
  res.v = _mm_div_pd(a.v, b.v);
}

static inline void simd_div_double2(simd_double2& a, simd_double2& b)
{
  simd_div_double2(a, a, b);
}

static inline void simd_sqrt_double2(simd_double2& res, simd_double2& a)
{
  res.v = _mm_sqrt_pd(a.v);
}

static inline void simd_sqrt_double2(simd_double2& a)
{
  simd_sqrt_double2(a, a);
}

static inline void simd_copy_double2(simd_double2& dst, simd_double2& a)
{
  _mm_storeu_pd(dst.n, a.v);
}

typedef union
{
  __m128i v;
  int8_t n[16];
} simd_char16;

static inline void simd_set_char16(simd_char16& res,
  int8_t a, int8_t b, int8_t c, int8_t d,
  int8_t e, int8_t f, int8_t g, int8_t h,
  int8_t i, int8_t j, int8_t k, int8_t l,
  int8_t m, int8_t n, int8_t o, int8_t p)
{
  res.v = _mm_set_epi8(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
}

static inline void simd_set_char16(simd_char16& res, int8_t v[16])
{
  res.v = _mm_set_epi8(
    v[0], v[1], v[2], v[3],
    v[4], v[5], v[6], v[7],
    v[8], v[9], v[10], v[11],
    v[12], v[13], v[14], v[15]
  );
}

static inline void simd_set_char16(simd_char16& res, int8_t a)
{
  res.v = _mm_set1_epi8(a);
}

static inline void simd_add_char16(simd_char16& res, simd_char16& a, simd_char16& b)
{
  res.v = _mm_add_epi8(a.v, b.v);
}

static inline void simd_add_char16(simd_char16& a, simd_char16& b)
{
  simd_add_char16(a, a, b);
}

static inline void simd_sub_char16(simd_char16& res, simd_char16& a, simd_char16& b)
{
  res.v = _mm_sub_epi8(a.v, b.v);
}

static inline void simd_sub_char16(simd_char16& a, simd_char16& b)
{
  simd_sub_char16(a, a, b);
}

typedef union
{
  __m128i v;
  int16_t n[8];
} simd_short8;

static inline void simd_set_char16(simd_short8& res,
  int16_t a, int16_t b, int16_t c, int16_t d,
  int16_t e, int16_t f, int16_t g, int16_t h)
{
  res.v = _mm_set_epi16(a, b, c, d, e, f, g, h);
}

static inline void simd_set_short8(simd_short8& res, int16_t v[8])
{
  res.v = _mm_set_epi16(
    v[0], v[1], v[2], v[3],
    v[4], v[5], v[6], v[7]
  );
}

static inline void simd_set_short8(simd_short8& res, int16_t a)
{
  res.v = _mm_set1_epi16(a);
}

static inline void simd_add_short8(simd_short8& res, simd_short8& a, simd_short8& b)
{
  res.v = _mm_add_epi16(a.v, b.v);
}

static inline void simd_add_short8(simd_short8& a, simd_short8& b)
{
  simd_add_short8(a, a, b);
}

static inline void simd_sub_short8(simd_short8& res, simd_short8& a, simd_short8& b)
{
  res.v = _mm_sub_epi16(a.v, b.v);
}

static inline void simd_sub_short8(simd_short8& a, simd_short8& b)
{
  simd_sub_short8(a, a, b);
}

static inline void simd_mul_short8(simd_short8& res, simd_short8& a, simd_short8& b)
{
  res.v = _mm_mulhi_epi16(a.v, b.v);
}

static inline void simd_mul_short8(simd_short8& a, simd_short8& b)
{
  simd_mul_short8(a, a, b);
}

typedef union
{
  __m128i v;
  int32_t n[4];
} simd_long4;

static inline void simd_set_long4(simd_long4& res,
  int32_t a, int32_t b, int32_t c, int32_t d)
{
  res.v = _mm_set_epi32(a, b, c, d);
}

static inline void simd_set_long4(simd_long4& res, int32_t v[4])
{
  res.v = _mm_set_epi32(v[0], v[1], v[2], v[3]);
}

static inline void simd_set_long4(simd_long4& res, int32_t a)
{
  res.v = _mm_set1_epi32(a);
}

static inline void simd_add_long4(simd_long4& res, simd_long4& a, simd_long4& b)
{
  res.v = _mm_add_epi32(a.v, b.v);
}

static inline void simd_add_long4(simd_long4& a, simd_long4& b)
{
  simd_add_long4(a, a, b);
}

static inline void simd_sub_long4(simd_long4& res, simd_long4& a, simd_long4& b)
{
  res.v = _mm_sub_epi32(a.v, b.v);
}

static inline void simd_sub_long4(simd_long4& a, simd_long4& b)
{
  simd_sub_long4(a, a, b);
}

static inline void simd_mul_long4(simd_long4& res, simd_long4& a, simd_long4& b)
{
  res.v = _mm_mul_epu32(a.v, b.v);
}

static inline void simd_mul_long4(simd_long4& a, simd_long4& b)
{
  simd_mul_long4(a, a, b);
}

typedef union
{
  __m128i v;
  __m64 n_m64[2];
  int64_t n[2];
} simd_llong2;

static inline void simd_set_llong2(simd_llong2& res, int64_t a, int64_t b)
{
  res.v = _mm_set_epi64(_mm_set_pi64x(a), _mm_set_pi64x(b));
}

static inline void simd_set_llong2(simd_llong2& res, int64_t v[2])
{
  res.v = _mm_set_epi64(_mm_set_pi64x(v[0]), _mm_set_pi64x(v[1]));
}

static inline void simd_set_llong2(simd_llong2& res, int64_t a)
{
  res.v = _mm_set1_epi64(_mm_set_pi64x(a));
}

static inline void simd_add_llong2(simd_llong2& res, simd_llong2& a, simd_llong2& b)
{
  res.v = _mm_add_epi64(a.v, b.v);
}

static inline void simd_add_llong2(simd_llong2& a, simd_llong2& b)
{
  simd_add_llong2(a, a, b);
}

static inline void simd_sub_llong2(simd_llong2& res, simd_llong2& a, simd_llong2& b)
{
  res.v = _mm_sub_epi64(a.v, b.v);
}

static inline void simd_sub_llong2(simd_llong2& a, simd_llong2& b)
{
  simd_sub_llong2(a, a, b);
}

static inline void simd_copy_llong2(simd_llong2& dst, simd_llong2& a)
{
  _mm_storeu_si64(dst.n, a.v);
}

#endif /* end of include guard: __SIMD_VECTOR_INTRINSICS_H__ */

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rad.hpp"
#include "core/bhatt_lod.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor.hpp"
#include "io/atomic_output.hpp"
#include "io/error.hpp"

#include <cuda_runtime.h>
#include <libdeflate.h>
#include <nlohmann/json.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lfs::io {

    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;

    namespace {

        // ============================================================================
        // Constants
        // ============================================================================

        constexpr uint32_t RAD_MAGIC = 0x30444152;       // "RAD0" in little-endian
        constexpr uint32_t RAD_CHUNK_MAGIC = 0x43444152; // "RADC" in little-endian
        constexpr uint32_t CHUNK_SIZE = 65536;           // Splats per chunk
        constexpr int GZ_LEVEL = 6;                      // Default gzip compression level
        constexpr float SH_C0 = 0.28209479177387814f;    // Degree-0 SH basis constant

        // SH coefficient count per degree: 0->0, 1->3, 2->8, 3->15
        constexpr int SH_COEFFS_FOR_DEGREE[] = {0, 3, 8, 15};

        // ============================================================================
        // Encoding Type Enums
        // ============================================================================

        enum class RadCenterEncoding : uint8_t {
            Auto = 0,
            F32 = 1,
            F32LeBytes = 2,
            F16 = 3,
            F16LeBytes = 4
        };

        enum class RadAlphaEncoding : uint8_t {
            Auto = 0,
            F32 = 1,
            F16 = 2,
            R8 = 3
        };

        enum class RadRgbEncoding : uint8_t {
            Auto = 0,
            F32 = 1,
            F16 = 2,
            R8 = 3,
            R8Delta = 4
        };

        enum class RadScalesEncoding : uint8_t {
            Auto = 0,
            F32 = 1,
            Ln0R8 = 2,
            LnF16 = 3
        };

        enum class RadOrientationEncoding : uint8_t {
            Auto = 0,
            F32 = 1,
            F16 = 2,
            Oct88R8 = 3
        };

        enum class RadShEncoding : uint8_t {
            Auto = 0,
            F32 = 1,
            F16 = 2,
            S8 = 3,
            S8Delta = 4
        };

        // ============================================================================
        // Property Names
        // ============================================================================

        constexpr const char* PROP_CENTER = "center";
        constexpr const char* PROP_ALPHA = "alpha";
        constexpr const char* PROP_RGB = "rgb";
        constexpr const char* PROP_SCALES = "scales";
        constexpr const char* PROP_ORIENTATION = "orientation";
        constexpr const char* PROP_SH1 = "sh1";
        constexpr const char* PROP_SH2 = "sh2";
        constexpr const char* PROP_SH3 = "sh3";
        constexpr const char* PROP_CHILD_COUNT = "child_count";
        constexpr const char* PROP_CHILD_START = "child_start";

        // ============================================================================
        // Utility Functions
        // ============================================================================

        // Write little-endian uint16_t
        inline void encode_u16(uint8_t* dst, uint16_t value) {
            dst[0] = static_cast<uint8_t>(value & 0xFF);
            dst[1] = static_cast<uint8_t>((value >> 8) & 0xFF);
        }

        // Write little-endian uint32_t
        inline void encode_u32(uint8_t* dst, uint32_t value) {
            dst[0] = static_cast<uint8_t>(value & 0xFF);
            dst[1] = static_cast<uint8_t>((value >> 8) & 0xFF);
            dst[2] = static_cast<uint8_t>((value >> 16) & 0xFF);
            dst[3] = static_cast<uint8_t>((value >> 24) & 0xFF);
        }

        // Write little-endian uint64_t
        inline void encode_u64(uint8_t* dst, uint64_t value) {
            dst[0] = static_cast<uint8_t>(value & 0xFF);
            dst[1] = static_cast<uint8_t>((value >> 8) & 0xFF);
            dst[2] = static_cast<uint8_t>((value >> 16) & 0xFF);
            dst[3] = static_cast<uint8_t>((value >> 24) & 0xFF);
            dst[4] = static_cast<uint8_t>((value >> 32) & 0xFF);
            dst[5] = static_cast<uint8_t>((value >> 40) & 0xFF);
            dst[6] = static_cast<uint8_t>((value >> 48) & 0xFF);
            dst[7] = static_cast<uint8_t>((value >> 56) & 0xFF);
        }

        // Read little-endian uint16_t
        inline uint16_t decode_u16(const uint8_t* src) {
            return static_cast<uint16_t>(src[0]) |
                   (static_cast<uint16_t>(src[1]) << 8);
        }

        // Read little-endian uint32_t
        inline uint32_t decode_u32(const uint8_t* src) {
            return static_cast<uint32_t>(src[0]) |
                   (static_cast<uint32_t>(src[1]) << 8) |
                   (static_cast<uint32_t>(src[2]) << 16) |
                   (static_cast<uint32_t>(src[3]) << 24);
        }

        // Read little-endian uint64_t
        inline uint64_t decode_u64(const uint8_t* src) {
            return static_cast<uint64_t>(src[0]) |
                   (static_cast<uint64_t>(src[1]) << 8) |
                   (static_cast<uint64_t>(src[2]) << 16) |
                   (static_cast<uint64_t>(src[3]) << 24) |
                   (static_cast<uint64_t>(src[4]) << 32) |
                   (static_cast<uint64_t>(src[5]) << 40) |
                   (static_cast<uint64_t>(src[6]) << 48) |
                   (static_cast<uint64_t>(src[7]) << 56);
        }

        // Pad size to 8-byte alignment
        inline size_t pad8(size_t size) {
            return (size + 7) & ~7;
        }

        // Padding bytes required to reach 8-byte alignment
        inline size_t pad8_len(size_t size) {
            return (8 - (size & 7)) & 7;
        }

        // ============================================================================
        // Half-Precision Float Conversion
        // ============================================================================

        // Convert float32 to float16 (IEEE 754)
        inline uint16_t float32_to_float16(float value) {
            uint32_t f32;
            std::memcpy(&f32, &value, sizeof(float));

            uint32_t sign = (f32 >> 31) & 0x1;
            uint32_t exponent = (f32 >> 23) & 0xFF;
            uint32_t mantissa = f32 & 0x7FFFFF;

            uint16_t f16;

            if (exponent == 0) {
                // Zero or subnormal - flush to zero
                f16 = static_cast<uint16_t>(sign << 15);
            } else if (exponent == 0xFF) {
                // Infinity or NaN
                f16 = static_cast<uint16_t>((sign << 15) | 0x7C00 | (mantissa >> 13));
            } else {
                // Normal number
                int32_t new_exp = static_cast<int32_t>(exponent) - 127 + 15;
                if (new_exp >= 31) {
                    // Overflow to infinity
                    f16 = static_cast<uint16_t>((sign << 15) | 0x7C00);
                } else if (new_exp <= 0) {
                    // Underflow to zero
                    f16 = static_cast<uint16_t>(sign << 15);
                } else {
                    uint32_t new_mantissa = mantissa >> 13;
                    // Round to nearest even
                    if ((mantissa & 0x1FFF) > 0x1000 ||
                        ((mantissa & 0x1FFF) == 0x1000 && (new_mantissa & 1))) {
                        new_mantissa++;
                    }
                    f16 = static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(new_exp) << 10) | new_mantissa);
                }
            }

            return f16;
        }

        // Convert float16 to float32
        inline float float16_to_float32(uint16_t value) {
            uint32_t sign = (value >> 15) & 0x1;
            uint32_t exponent = (value >> 10) & 0x1F;
            uint32_t mantissa = value & 0x3FF;

            uint32_t f32;

            if (exponent == 0) {
                // Zero or subnormal
                if (mantissa == 0) {
                    f32 = sign << 31;
                } else {
                    // Subnormal - normalize it
                    int shift = 10 - static_cast<int>(std::log2(mantissa));
                    exponent = 1 - shift;
                    mantissa <<= shift;
                    f32 = (sign << 31) | ((exponent + 127 - 15) << 23) | ((mantissa & 0x3FF) << 13);
                }
            } else if (exponent == 0x1F) {
                // Infinity or NaN
                f32 = (sign << 31) | (0xFF << 23) | (mantissa << 13);
            } else {
                // Normal number
                f32 = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
            }

            float result;
            std::memcpy(&result, &f32, sizeof(float));
            return result;
        }

        // ============================================================================
        // RAD Compression/Decompression
        // ============================================================================

        struct TlsDeflateCompressor {
            libdeflate_compressor* handle = nullptr;
            int level = -1;

            ~TlsDeflateCompressor() {
                if (handle != nullptr) {
                    libdeflate_free_compressor(handle);
                }
            }

            libdeflate_compressor* get(int requested_level) {
                if (handle == nullptr || level != requested_level) {
                    if (handle != nullptr) {
                        libdeflate_free_compressor(handle);
                    }
                    handle = libdeflate_alloc_compressor(requested_level);
                    level = requested_level;
                }
                return handle;
            }
        };

        struct TlsDeflateDecompressor {
            libdeflate_decompressor* handle = nullptr;

            ~TlsDeflateDecompressor() {
                if (handle != nullptr) {
                    libdeflate_free_decompressor(handle);
                }
            }

            libdeflate_decompressor* get() {
                if (handle == nullptr) {
                    handle = libdeflate_alloc_decompressor();
                }
                return handle;
            }
        };

        thread_local TlsDeflateCompressor tls_compressor;
        thread_local TlsDeflateDecompressor tls_decompressor;

        // Reference RAD writers mark compression as "gz" but emit raw DEFLATE streams
        // (without gzip/zlib wrapper bytes).
        std::vector<uint8_t> rad_compress_zlib(const uint8_t* data, size_t size, int level = GZ_LEVEL) {
            if (size == 0) {
                return {};
            }

            z_stream strm{};
            const int init_ret = deflateInit2(&strm, level, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY);
            if (init_ret != Z_OK) {
                LOG_ERROR("rad_compress: deflateInit2 failed (ret={}, level={})", init_ret, level);
                return {};
            }

            strm.next_in = const_cast<Bytef*>(data);
            strm.avail_in = static_cast<uInt>(size);

            std::vector<uint8_t> output;
            output.reserve(deflateBound(&strm, static_cast<uLong>(size)));

            const size_t chunk_size = 65536;
            std::vector<uint8_t> chunk(chunk_size);
            bool success = false;

            while (true) {
                strm.next_out = chunk.data();
                strm.avail_out = static_cast<uInt>(chunk.size());

                int ret = deflate(&strm, Z_FINISH);
                if (ret != Z_OK && ret != Z_STREAM_END) {
                    LOG_ERROR("rad_compress: deflate failed with error {}", ret);
                    break;
                }

                size_t have = chunk.size() - strm.avail_out;
                output.insert(output.end(), chunk.begin(), chunk.begin() + have);

                if (ret == Z_STREAM_END) {
                    success = true;
                    break;
                }
            }

            deflateEnd(&strm);

            if (!success) {
                LOG_ERROR("rad_compress: compression failed");
                return {};
            }

            return output;
        }

        std::vector<uint8_t> rad_compress(const uint8_t* data, size_t size, int level = GZ_LEVEL) {
            if (size == 0) {
                return {};
            }

            const int effective_level = std::clamp(level == Z_DEFAULT_COMPRESSION ? GZ_LEVEL : level, 0, 9);
            libdeflate_compressor* compressor = tls_compressor.get(effective_level);
            if (compressor == nullptr) {
                return rad_compress_zlib(data, size, level);
            }

            std::vector<uint8_t> output(libdeflate_deflate_compress_bound(compressor, size));
            const size_t written = libdeflate_deflate_compress(compressor, data, size, output.data(), output.size());
            if (written == 0) {
                return rad_compress_zlib(data, size, level);
            }
            output.resize(written);
            return output;
        }

        std::optional<std::vector<uint8_t>> inflate_with_window_bits(const uint8_t* data, size_t size, int window_bits) {
            z_stream strm = {};
            strm.zalloc = Z_NULL;
            strm.zfree = Z_NULL;
            strm.opaque = Z_NULL;
            strm.avail_in = static_cast<uInt>(size);
            strm.next_in = const_cast<Bytef*>(data);

            int ret = inflateInit2(&strm, window_bits);
            if (ret != Z_OK) {
                return std::nullopt;
            }

            std::vector<uint8_t> output;
            const size_t chunk_size = 65536;
            std::vector<uint8_t> chunk(chunk_size);
            bool success = false;

            do {
                strm.avail_out = static_cast<uInt>(chunk.size());
                strm.next_out = chunk.data();

                ret = inflate(&strm, Z_NO_FLUSH);
                if (ret == Z_STREAM_ERROR || ret == Z_DATA_ERROR || ret == Z_MEM_ERROR) {
                    inflateEnd(&strm);
                    return std::nullopt;
                }

                size_t have = chunk.size() - strm.avail_out;
                output.insert(output.end(), chunk.begin(), chunk.begin() + have);
                if (ret == Z_STREAM_END) {
                    success = true;
                }
            } while (ret != Z_STREAM_END);

            inflateEnd(&strm);
            if (!success) {
                return std::nullopt;
            }
            return output;
        }

        std::vector<uint8_t> rad_decompress(const uint8_t* data, size_t size) {
            // Match reference first.
            if (auto raw = inflate_with_window_bits(data, size, -15)) {
                return std::move(*raw);
            }
            // Backward compatibility for older local files.
            if (auto gzip = inflate_with_window_bits(data, size, 15 + 16)) {
                return std::move(*gzip);
            }
            // Accept zlib wrapper as a permissive fallback.
            if (auto zlib = inflate_with_window_bits(data, size, 15)) {
                return std::move(*zlib);
            }
            return {};
        }

        // Fast one-shot decompression when the decoded size is known from the
        // property layout. Falls back to streaming zlib for size mismatches and
        // wrapped/legacy streams.
        std::vector<uint8_t> rad_decompress_sized(const uint8_t* data, size_t size, size_t expected_bytes) {
            if (expected_bytes > 0 && size > 0) {
                if (libdeflate_decompressor* decompressor = tls_decompressor.get()) {
                    std::vector<uint8_t> output(expected_bytes);
                    size_t actual = 0;
                    if (libdeflate_deflate_decompress(decompressor, data, size,
                                                      output.data(), output.size(), &actual) == LIBDEFLATE_SUCCESS &&
                        actual == expected_bytes) {
                        return output;
                    }
                    if (libdeflate_gzip_decompress(decompressor, data, size,
                                                   output.data(), output.size(), &actual) == LIBDEFLATE_SUCCESS &&
                        actual == expected_bytes) {
                        return output;
                    }
                    if (libdeflate_zlib_decompress(decompressor, data, size,
                                                   output.data(), output.size(), &actual) == LIBDEFLATE_SUCCESS &&
                        actual == expected_bytes) {
                        return output;
                    }
                }
            }
            return rad_decompress(data, size);
        }

        // Decoded byte count implied by a property's name + encoding, or 0 when unknown.
        size_t rad_property_decoded_bytes(const std::string& property, const std::string& encoding, size_t count) {
            size_t dims = 0;
            if (property == PROP_CENTER || property == PROP_RGB || property == PROP_SCALES) {
                dims = 3;
            } else if (property == PROP_ALPHA || property == PROP_CHILD_COUNT || property == PROP_CHILD_START) {
                dims = 1;
            } else if (property == PROP_ORIENTATION) {
                return encoding == "oct88r8" ? count * 3 : count * 3 * (encoding == "f16" ? 2 : 4);
            } else if (property == PROP_SH1) {
                dims = 9;
            } else if (property == PROP_SH2) {
                dims = 15;
            } else if (property == PROP_SH3) {
                dims = 21;
            } else if (property.find(PROP_CENTER) == 0 || property.find(PROP_RGB) == 0 ||
                       property.find(PROP_SCALES) == 0 || property.find("sh") == 0) {
                dims = 1;
            } else {
                return 0;
            }

            size_t element_bytes = 0;
            if (encoding == "f32" || encoding == "f32_lebytes" || encoding == "u32") {
                element_bytes = 4;
            } else if (encoding == "f16" || encoding == "f16_lebytes" || encoding == "ln_f16" || encoding == "u16") {
                element_bytes = 2;
            } else if (encoding == "r8" || encoding == "r8_delta" || encoding == "s8" ||
                       encoding == "s8_delta" || encoding == "ln_0r8") {
                element_bytes = 1;
            } else {
                return 0;
            }
            return count * dims * element_bytes;
        }

        // ============================================================================
        // Encoding Functions
        // ============================================================================

        // Encode interleaved [count, dims] floats into dimension-major f32 bytes.
        std::vector<uint8_t> encode_f32(const float* data, size_t dims, size_t count) {
            std::vector<uint8_t> result(count * dims * 4);
            size_t out_idx = 0;

            for (size_t d = 0; d < dims; ++d) {
                size_t index = d;
                for (size_t i = 0; i < count; ++i) {
                    float v = data[index];
                    const auto* bytes = reinterpret_cast<const uint8_t*>(&v);
                    result[out_idx++] = bytes[0];
                    result[out_idx++] = bytes[1];
                    result[out_idx++] = bytes[2];
                    result[out_idx++] = bytes[3];
                    index += dims;
                }
            }
            return result;
        }

        void decode_f32(const uint8_t* encoded, float* output, size_t dims, size_t count) {
            for (size_t i = 0; i < count; ++i) {
                for (size_t d = 0; d < dims; ++d) {
                    const size_t src = (d * count + i) * 4;
                    uint32_t bits = decode_u32(&encoded[src]);
                    std::memcpy(&output[i * dims + d], &bits, sizeof(float));
                }
            }
        }

        // Encode interleaved [count, dims] floats into byte-interleaved little-endian blocks.
        std::vector<uint8_t> encode_f32_lebytes(const float* data, size_t dims, size_t count) {
            std::vector<uint8_t> result(count * dims * 4);
            const size_t stride = count * dims;
            for (size_t b = 0; b < 4; ++b) {
                for (size_t d = 0; d < dims; ++d) {
                    size_t index = d;
                    for (size_t i = 0; i < count; ++i) {
                        const auto* bytes = reinterpret_cast<const uint8_t*>(&data[index]);
                        result[b * stride + d * count + i] = bytes[b];
                        index += dims;
                    }
                }
            }
            return result;
        }

        void decode_f32_lebytes(const uint8_t* encoded, float* output, size_t dims, size_t count) {
            const size_t stride = count * dims;
            for (size_t i = 0; i < count; ++i) {
                for (size_t d = 0; d < dims; ++d) {
                    uint8_t bytes[4];
                    for (size_t b = 0; b < 4; ++b) {
                        bytes[b] = encoded[b * stride + d * count + i];
                    }
                    float v;
                    std::memcpy(&v, bytes, sizeof(float));
                    output[i * dims + d] = v;
                }
            }
        }

        std::vector<uint8_t> encode_f16(const float* data, size_t dims, size_t count) {
            std::vector<uint8_t> result(count * dims * 2);
            size_t out_idx = 0;

            for (size_t d = 0; d < dims; ++d) {
                size_t index = d;
                for (size_t i = 0; i < count; ++i) {
                    uint16_t v = float32_to_float16(data[index]);
                    result[out_idx++] = static_cast<uint8_t>(v & 0xFF);
                    result[out_idx++] = static_cast<uint8_t>((v >> 8) & 0xFF);
                    index += dims;
                }
            }
            return result;
        }

        void decode_f16(const uint8_t* encoded, float* output, size_t dims, size_t count) {
            for (size_t i = 0; i < count; ++i) {
                for (size_t d = 0; d < dims; ++d) {
                    const size_t src = (d * count + i) * 2;
                    uint16_t f16 = decode_u16(&encoded[src]);
                    output[i * dims + d] = float16_to_float32(f16);
                }
            }
        }

        std::vector<uint8_t> encode_f16_lebytes(const float* data, size_t dims, size_t count) {
            std::vector<uint8_t> result(count * dims * 2);
            const size_t stride = count * dims;
            for (size_t b = 0; b < 2; ++b) {
                for (size_t d = 0; d < dims; ++d) {
                    size_t index = d;
                    for (size_t i = 0; i < count; ++i) {
                        uint16_t f16 = float32_to_float16(data[index]);
                        result[b * stride + d * count + i] = (b == 0)
                                                                 ? static_cast<uint8_t>(f16 & 0xFF)
                                                                 : static_cast<uint8_t>((f16 >> 8) & 0xFF);
                        index += dims;
                    }
                }
            }
            return result;
        }

        void decode_f16_lebytes(const uint8_t* encoded, float* output, size_t dims, size_t count) {
            const size_t stride = count * dims;
            for (size_t i = 0; i < count; ++i) {
                for (size_t d = 0; d < dims; ++d) {
                    uint8_t b0 = encoded[d * count + i];
                    uint8_t b1 = encoded[stride + d * count + i];
                    output[i * dims + d] = float16_to_float32(static_cast<uint16_t>(b0 | (b1 << 8)));
                }
            }
        }

        // Encode f32 to R8 (8-bit quantized with min/max)
        struct R8Result {
            std::vector<uint8_t> data;
            float min_val;
            float max_val;
        };

        R8Result encode_r8(const float* data, size_t dims, size_t count, std::optional<float> forced_min = std::nullopt,
                           std::optional<float> forced_max = std::nullopt) {
            if (count == 0 || dims == 0) {
                return {{}, 0.0f, 0.0f};
            }

            float min_val = forced_min.value_or(data[0]);
            float max_val = forced_max.value_or(data[0]);
            if (!forced_min.has_value() || !forced_max.has_value()) {
                for (size_t i = 0; i < count * dims; ++i) {
                    min_val = std::min(min_val, data[i]);
                    max_val = std::max(max_val, data[i]);
                }
            }

            float range = max_val - min_val;
            if (range < 1e-7f) {
                range = 1e-7f;
            }

            std::vector<uint8_t> result(count * dims);
            size_t out_idx = 0;
            for (size_t d = 0; d < dims; ++d) {
                size_t index = d;
                for (size_t i = 0; i < count; ++i) {
                    float normalized = (data[index] - min_val) / range;
                    result[out_idx++] = static_cast<uint8_t>(std::clamp(std::round(normalized * 255.0f), 0.0f, 255.0f));
                    index += dims;
                }
            }

            return {result, min_val, max_val};
        }

        // Decode R8 to f32
        void decode_r8(const uint8_t* encoded, float* output, size_t dims, size_t count, float min_val, float max_val) {
            float range = max_val - min_val;
            for (size_t i = 0; i < count; ++i) {
                for (size_t d = 0; d < dims; ++d) {
                    const size_t src = d * count + i;
                    output[i * dims + d] = min_val + (static_cast<float>(encoded[src]) / 255.0f) * range;
                }
            }
        }

        // Encode f32 to R8 with delta encoding
        struct R8DeltaResult {
            std::vector<uint8_t> data;
            float min_val;
            float max_val;
        };

        R8DeltaResult encode_r8_delta(const float* data, size_t dims, size_t count,
                                      std::optional<float> forced_min = std::nullopt,
                                      std::optional<float> forced_max = std::nullopt) {
            if (count == 0 || dims == 0) {
                return {{}, 0.0f, 0.0f};
            }

            auto base_quant = encode_r8(data, dims, count, forced_min, forced_max);
            std::vector<uint8_t> result;
            result.reserve(base_quant.data.size());
            for (size_t d = 0; d < dims; ++d) {
                uint8_t last = 0;
                for (size_t i = 0; i < count; ++i) {
                    const uint8_t value = base_quant.data[d * count + i];
                    result.push_back(static_cast<uint8_t>(value - last));
                    last = value;
                }
            }
            return {result, base_quant.min_val, base_quant.max_val};
        }

        // Decode R8 delta to f32
        void decode_r8_delta(const uint8_t* encoded, float* output, size_t dims, size_t count, float min_val, float max_val) {
            std::vector<uint8_t> quantized(count * dims, 0);
            for (size_t d = 0; d < dims; ++d) {
                uint8_t last = 0;
                for (size_t i = 0; i < count; ++i) {
                    const size_t idx = d * count + i;
                    last = static_cast<uint8_t>(last + encoded[idx]);
                    quantized[idx] = last;
                }
            }
            decode_r8(quantized.data(), output, dims, count, min_val, max_val);
        }

        // Encode f32 to S8 (signed 8-bit for SH coefficients)
        struct S8Result {
            std::vector<int8_t> data;
            float max_abs;
        };

        S8Result encode_s8(const float* data, size_t dims, size_t count, std::optional<float> forced_max = std::nullopt) {
            float max_val = 0.0f;
            if (forced_max.has_value()) {
                max_val = std::max(1e-6f, std::abs(forced_max.value()));
            } else {
                for (size_t i = 0; i < count * dims; ++i) {
                    max_val = std::max(max_val, std::abs(data[i]));
                }
                max_val = std::max(max_val, 1e-6f);
            }

            std::vector<int8_t> result(count * dims);
            size_t out_idx = 0;
            for (size_t d = 0; d < dims; ++d) {
                size_t index = d;
                for (size_t i = 0; i < count; ++i) {
                    float scaled = data[index] / max_val * 127.0f;
                    result[out_idx++] = static_cast<int8_t>(std::clamp(std::round(scaled), -127.0f, 127.0f));
                    index += dims;
                }
            }

            return {result, max_val};
        }

        // Decode S8 to f32
        void decode_s8(const int8_t* encoded, float* output, size_t dims, size_t count, float max_abs) {
            for (size_t i = 0; i < count; ++i) {
                for (size_t d = 0; d < dims; ++d) {
                    const size_t src = d * count + i;
                    output[i * dims + d] = (static_cast<float>(encoded[src]) / 127.0f) * max_abs;
                }
            }
        }

        // Encode f32 to S8 with delta encoding
        struct S8DeltaResult {
            std::vector<int8_t> data;
            float max_abs;
        };

        S8DeltaResult encode_s8_delta(const float* data, size_t dims, size_t count, std::optional<float> forced_max = std::nullopt) {
            auto base_quant = encode_s8(data, dims, count, forced_max);
            std::vector<int8_t> result;
            result.reserve(base_quant.data.size());
            for (size_t d = 0; d < dims; ++d) {
                uint8_t last = 0;
                for (size_t i = 0; i < count; ++i) {
                    const uint8_t value = static_cast<uint8_t>(base_quant.data[d * count + i]);
                    result.push_back(static_cast<int8_t>(value - last));
                    last = value;
                }
            }
            return {result, base_quant.max_abs};
        }

        // Decode S8 delta to f32
        void decode_s8_delta(const int8_t* encoded, float* output, size_t dims, size_t count, float max_abs) {
            std::vector<int8_t> quantized(count * dims, 0);
            for (size_t d = 0; d < dims; ++d) {
                uint8_t last = 0;
                for (size_t i = 0; i < count; ++i) {
                    const size_t idx = d * count + i;
                    last = static_cast<uint8_t>(last + static_cast<uint8_t>(encoded[idx]));
                    quantized[idx] = static_cast<int8_t>(last);
                }
            }
            decode_s8(quantized.data(), output, dims, count, max_abs);
        }

        // Encode scales to log-space 8-bit with zero handling
        // Algorithm: scale -> ln(scale) -> quantize with zero handling
        // Value 0 is reserved for zero scales, values 1-255 encode ln(scale) in [ln_min, ln_max]
        struct Ln0R8Result {
            std::vector<uint8_t> data;
            float min_val;
            float max_val;
        };

        Ln0R8Result encode_ln_0r8(const float* data, size_t dims, size_t count) {
            // First pass: compute ln of all positive scales and find range
            std::vector<float> log_values;
            log_values.reserve(count * dims);
            float ln_min = std::numeric_limits<float>::infinity();
            float ln_max = -std::numeric_limits<float>::infinity();

            for (size_t i = 0; i < count * dims; ++i) {
                if (data[i] > 0.0f) {
                    float ln_val = std::log(data[i]);
                    log_values.push_back(ln_val);
                    ln_min = std::min(ln_min, ln_val);
                    ln_max = std::max(ln_max, ln_val);
                } else {
                    log_values.push_back(-std::numeric_limits<float>::infinity()); // Marker for zero
                }
            }

            // Handle edge case: all zeros or single value
            if (!std::isfinite(ln_min) || !std::isfinite(ln_max) || ln_max - ln_min < 1e-7f) {
                ln_min = -10.0f; // Default ~exp(-10) = 4.5e-5
                ln_max = 2.0f;   // Default exp(2) = 7.4
            }

            // Compute ln_zero threshold (scales below this encode to 0)
            float ln_zero = ln_min - 1.0f;

            std::vector<uint8_t> result;
            result.reserve(count * dims);
            for (size_t d = 0; d < dims; ++d) {
                size_t index = d;
                for (size_t i = 0; i < count; ++i) {
                    float scale = data[index];
                    uint8_t encoded;
                    if (scale <= 0.0f) {
                        encoded = 0;
                    } else {
                        float ln_scale = std::log(scale);
                        if (ln_scale <= ln_zero) {
                            encoded = 0;
                        } else {
                            float normalized = (ln_scale - ln_min) / (ln_max - ln_min) * 254.0f;
                            encoded = static_cast<uint8_t>(std::clamp(std::round(normalized), 0.0f, 254.0f)) + 1;
                        }
                    }
                    result.push_back(encoded);
                    index += dims;
                }
            }

            return {result, ln_min, ln_max};
        }

        // Decode log-space 8-bit to scales
        // Value 0 decodes to 0, values 1-255 decode to exp(ln) in [ln_min, ln_max]
        void decode_ln_0r8(const uint8_t* encoded, float* output, size_t dims, size_t count, float ln_min, float ln_max) {
            for (size_t i = 0; i < count; ++i) {
                for (size_t d = 0; d < dims; ++d) {
                    const size_t src = d * count + i;
                    uint8_t value = encoded[src];
                    if (value == 0) {
                        output[i * dims + d] = 0.0f;
                    } else {
                        float ln_scale = ln_min + (value - 1) * (ln_max - ln_min) / 254.0f;
                        output[i * dims + d] = std::exp(ln_scale);
                    }
                }
            }
        }

        // Encode scales to log-space f16
        std::vector<uint8_t> encode_ln_f16(const float* data, size_t dims, size_t count) {
            std::vector<float> log_data(count * dims);
            for (size_t i = 0; i < count * dims; ++i) {
                log_data[i] = std::log(data[i]);
            }
            return encode_f16(log_data.data(), dims, count);
        }

        // Decode log-space f16 to scales
        void decode_ln_f16(const uint8_t* encoded, float* output, size_t dims, size_t count) {
            decode_f16(encoded, output, dims, count);
            for (size_t i = 0; i < count * dims; ++i) {
                output[i] = std::exp(output[i]);
            }
        }

        // Encode quaternion to octahedral 3-byte representation using axis-angle encoding
        // Returns 3 bytes per quaternion:
        //   - 2 bytes: rotation axis encoded with octahedral projection
        //   - 1 byte: rotation angle (theta) quantized to 8 bits
        //
        // Algorithm:
        //   1. Ensure positive w (if w < 0, negate all components)
        //   2. Extract angle: theta = 2 * acos(w)
        //   3. Extract axis: if sin(theta/2) > epsilon, axis = (x,y,z) / sin(theta/2), else use (1,0,0)
        //   4. Encode axis to octahedral (2 bytes)
        //   5. Encode angle to 1 byte: theta / PI * 255
        std::vector<uint8_t> encode_quat_oct88r8(const float* quats, size_t count, bool input_wxyz = false) {
            std::vector<uint8_t> result(count * 3);
            constexpr float PI = 3.14159265358979323846f;
            constexpr float EPSILON = 1e-6f;

            for (size_t i = 0; i < count; ++i) {
                float x = input_wxyz ? quats[i * 4 + 1] : quats[i * 4 + 0];
                float y = input_wxyz ? quats[i * 4 + 2] : quats[i * 4 + 1];
                float z = input_wxyz ? quats[i * 4 + 3] : quats[i * 4 + 2];
                float w = input_wxyz ? quats[i * 4 + 0] : quats[i * 4 + 3];

                // Normalize
                float len = std::sqrt(x * x + y * y + z * z + w * w);
                if (len > 0.0f) {
                    x /= len;
                    y /= len;
                    z /= len;
                    w /= len;
                }

                // Ensure positive w for consistency
                if (w < 0.0f) {
                    x = -x;
                    y = -y;
                    z = -z;
                    w = -w;
                }

                // Extract angle: theta = 2 * acos(w)
                float theta = 2.0f * std::acos(std::clamp(w, -1.0f, 1.0f));

                // Extract axis
                float sin_half_theta = std::sin(theta * 0.5f);
                float axis_x, axis_y, axis_z;
                if (sin_half_theta > EPSILON) {
                    axis_x = x / sin_half_theta;
                    axis_y = y / sin_half_theta;
                    axis_z = z / sin_half_theta;
                } else {
                    // Near identity rotation, use default axis
                    axis_x = 1.0f;
                    axis_y = 0.0f;
                    axis_z = 0.0f;
                }

                // Normalize axis
                float axis_len = std::sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z);
                if (axis_len > 0.0f) {
                    axis_x /= axis_len;
                    axis_y /= axis_len;
                    axis_z /= axis_len;
                }

                // Octahedral encoding of axis (project to octahedron then to square)
                float abs_x = std::abs(axis_x);
                float abs_y = std::abs(axis_y);
                float abs_z = std::abs(axis_z);

                float oct_x, oct_y, oct_z;
                if (abs_x + abs_y + abs_z > 0.0f) {
                    float inv_sum = 1.0f / (abs_x + abs_y + abs_z);
                    oct_x = axis_x * inv_sum;
                    oct_y = axis_y * inv_sum;
                    oct_z = axis_z * inv_sum;
                } else {
                    oct_x = axis_x;
                    oct_y = axis_y;
                    oct_z = axis_z;
                }

                // Fold to upper hemisphere if needed
                if (oct_z < 0.0f) {
                    float temp_x = oct_x;
                    oct_x = (1.0f - std::abs(oct_y)) * (oct_x >= 0.0f ? 1.0f : -1.0f);
                    oct_y = (1.0f - std::abs(temp_x)) * (oct_y >= 0.0f ? 1.0f : -1.0f);
                }

                // Map from [-1, 1] to [0, 255] for axis
                result[i * 3 + 0] = static_cast<uint8_t>(std::clamp((oct_x + 1.0f) * 0.5f * 255.0f, 0.0f, 255.0f));
                result[i * 3 + 1] = static_cast<uint8_t>(std::clamp((oct_y + 1.0f) * 0.5f * 255.0f, 0.0f, 255.0f));

                // Encode angle: theta / PI * 255
                result[i * 3 + 2] = static_cast<uint8_t>(std::clamp(theta / PI * 255.0f, 0.0f, 255.0f));
            }

            return result;
        }

        // Decode octahedral 3-byte to quaternion using axis-angle decoding
        // Input: 3 bytes per quaternion (2 bytes axis + 1 byte angle)
        // Algorithm:
        //   1. Decode axis from octahedral (2 bytes)
        //   2. Decode angle from 1 byte: theta = value / 255.0 * PI
        //   3. Reconstruct quaternion: w = cos(theta/2), (x,y,z) = axis * sin(theta/2)
        void decode_quat_oct88r8(const uint8_t* encoded, float* quats, size_t count) {
            constexpr float PI = 3.14159265358979323846f;

            for (size_t i = 0; i < count; ++i) {
                // Map from [0, 255] to [-1, 1] for octahedral coordinates
                float oct_x = (static_cast<float>(encoded[i * 3 + 0]) / 255.0f) * 2.0f - 1.0f;
                float oct_y = (static_cast<float>(encoded[i * 3 + 1]) / 255.0f) * 2.0f - 1.0f;

                // Unfold from lower hemisphere when the stored oct_z was negative
                float oct_z = 1.0f - std::abs(oct_x) - std::abs(oct_y);
                if (oct_z < 0.0f) {
                    float temp_x = oct_x;
                    oct_x = (1.0f - std::abs(oct_y)) * (oct_x >= 0.0f ? 1.0f : -1.0f);
                    oct_y = (1.0f - std::abs(temp_x)) * (oct_y >= 0.0f ? 1.0f : -1.0f);
                }

                // Project from octahedron to sphere (normalize)
                float axis_x = oct_x;
                float axis_y = oct_y;
                float axis_z = oct_z;
                float len = std::sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z);
                if (len > 0.0f) {
                    axis_x /= len;
                    axis_y /= len;
                    axis_z /= len;
                }

                // Decode angle: theta = value / 255.0 * PI
                float theta = (static_cast<float>(encoded[i * 3 + 2]) / 255.0f) * PI;

                // Reconstruct quaternion from axis-angle
                float half_theta = theta * 0.5f;
                float sin_half_theta = std::sin(half_theta);
                float cos_half_theta = std::cos(half_theta);

                float x = axis_x * sin_half_theta;
                float y = axis_y * sin_half_theta;
                float z = axis_z * sin_half_theta;
                float w = cos_half_theta;

                // Normalize
                float q_len = std::sqrt(x * x + y * y + z * z + w * w);
                if (q_len > 0.0f) {
                    x /= q_len;
                    y /= q_len;
                    z /= q_len;
                    w /= q_len;
                }

                quats[i * 4 + 0] = x;
                quats[i * 4 + 1] = y;
                quats[i * 4 + 2] = z;
                quats[i * 4 + 3] = w;
            }
        }

        // ============================================================================
        // Metadata Structures
        // ============================================================================

        struct RadChunkProperty {
            uint64_t offset = 0;
            uint64_t bytes = 0;
            std::string property;
            std::string encoding;
            std::optional<std::string> compression;
            std::optional<float> min_val;
            std::optional<float> max_val;
            std::optional<float> base;
            std::optional<float> scale;

            nlohmann::json to_json() const {
                nlohmann::json j;
                j["offset"] = offset;
                j["bytes"] = bytes;
                j["property"] = property;
                j["encoding"] = encoding;
                if (compression.has_value())
                    j["compression"] = compression.value();
                if (min_val.has_value())
                    j["min"] = min_val.value();
                if (max_val.has_value())
                    j["max"] = max_val.value();
                if (base.has_value())
                    j["base"] = base.value();
                if (scale.has_value())
                    j["scale"] = scale.value();
                return j;
            }

            static RadChunkProperty from_json(const nlohmann::json& j) {
                RadChunkProperty prop;
                prop.offset = j.at("offset").get<uint64_t>();
                prop.bytes = j.at("bytes").get<uint64_t>();
                prop.property = j.at("property").get<std::string>();
                prop.encoding = j.at("encoding").get<std::string>();
                if (j.contains("compression"))
                    prop.compression = j.at("compression").get<std::string>();
                if (j.contains("min"))
                    prop.min_val = j.at("min").get<float>();
                if (j.contains("max"))
                    prop.max_val = j.at("max").get<float>();
                if (j.contains("base"))
                    prop.base = j.at("base").get<float>();
                if (j.contains("scale"))
                    prop.scale = j.at("scale").get<float>();
                return prop;
            }
        };

        struct RadChunkMeta {
            uint32_t version = 1;
            uint64_t base = 0;
            uint64_t count = 0;
            uint64_t payload_bytes = 0;
            int max_sh = 0;
            bool lod_tree = false;
            std::optional<nlohmann::json> splat_encoding;
            std::vector<RadChunkProperty> properties;

            nlohmann::json to_json() const {
                nlohmann::json j;
                j["version"] = version;
                j["base"] = base;
                j["count"] = count;
                j["payloadBytes"] = payload_bytes; // camelCase
                if (max_sh > 0)
                    j["maxSh"] = max_sh; // camelCase, optional
                if (lod_tree)
                    j["lodTree"] = lod_tree; // camelCase, optional
                if (splat_encoding.has_value())
                    j["splatEncoding"] = splat_encoding.value(); // camelCase

                nlohmann::json props = nlohmann::json::array();
                for (const auto& prop : properties) {
                    props.push_back(prop.to_json());
                }
                j["properties"] = props;
                return j;
            }

            static RadChunkMeta from_json(const nlohmann::json& j) {
                RadChunkMeta meta;
                meta.version = j.at("version").get<uint32_t>();
                meta.base = j.at("base").get<uint64_t>();
                meta.count = j.at("count").get<uint64_t>();
                meta.payload_bytes = j.at("payloadBytes").get<uint64_t>(); // camelCase
                if (j.contains("maxSh"))
                    meta.max_sh = j.at("maxSh").get<int>();
                if (j.contains("lodTree"))
                    meta.lod_tree = j.at("lodTree").get<bool>();
                if (j.contains("splatEncoding")) {
                    meta.splat_encoding = j.at("splatEncoding");
                }
                for (const auto& prop_json : j.at("properties")) {
                    meta.properties.push_back(RadChunkProperty::from_json(prop_json));
                }
                return meta;
            }
        };

        struct RadChunkRange {
            uint64_t offset = 0;
            uint64_t bytes = 0;
            std::optional<uint64_t> base;
            std::optional<uint64_t> count;
            std::optional<std::string> filename;

            nlohmann::json to_json() const {
                nlohmann::json j;
                j["offset"] = offset;
                j["bytes"] = bytes;
                if (base.has_value())
                    j["base"] = base.value();
                if (count.has_value())
                    j["count"] = count.value();
                if (filename.has_value())
                    j["filename"] = filename.value();
                return j;
            }

            static RadChunkRange from_json(const nlohmann::json& j) {
                RadChunkRange range;
                range.offset = j.at("offset").get<uint64_t>();
                range.bytes = j.at("bytes").get<uint64_t>();
                if (j.contains("base"))
                    range.base = j.at("base").get<uint64_t>();
                if (j.contains("count"))
                    range.count = j.at("count").get<uint64_t>();
                if (j.contains("filename"))
                    range.filename = j.at("filename").get<std::string>();
                return range;
            }
        };

        struct RadMeta {
            uint32_t version = 1;
            std::string type = "gsplat";
            uint64_t count = 0;                 // Changed from uint32_t
            std::optional<int> max_sh;          // Changed to optional
            std::optional<bool> lod_tree;       // Changed to optional
            std::optional<uint32_t> chunk_size; // Changed to optional
            uint64_t all_chunk_bytes = 0;
            std::vector<RadChunkRange> chunks; // Changed from RadChunkMeta
            std::optional<nlohmann::json> splat_encoding;
            std::optional<uint32_t> sh_code_count; // Added missing field
            std::optional<std::string> comment;

            nlohmann::json to_json() const {
                nlohmann::json j;
                j["version"] = version;
                j["type"] = type;
                j["count"] = count;
                if (max_sh.has_value())
                    j["maxSh"] = max_sh.value(); // camelCase
                if (lod_tree.has_value())
                    j["lodTree"] = lod_tree.value(); // camelCase
                if (chunk_size.has_value())
                    j["chunkSize"] = chunk_size.value(); // camelCase
                j["allChunkBytes"] = all_chunk_bytes;    // camelCase

                nlohmann::json chunks_json = nlohmann::json::array();
                for (const auto& chunk : chunks) {
                    chunks_json.push_back(chunk.to_json());
                }
                j["chunks"] = chunks_json;

                if (splat_encoding.has_value()) {
                    j["splatEncoding"] = splat_encoding.value(); // camelCase
                }
                if (sh_code_count.has_value()) {
                    j["shCodeCount"] = sh_code_count.value(); // camelCase
                }
                if (comment.has_value()) {
                    j["comment"] = comment.value();
                }
                return j;
            }

            static RadMeta from_json(const nlohmann::json& j) {
                RadMeta meta;
                meta.version = j.at("version").get<uint32_t>();
                meta.type = j.at("type").get<std::string>();
                meta.count = j.at("count").get<uint64_t>();
                if (j.contains("maxSh"))
                    meta.max_sh = j.at("maxSh").get<int>();
                if (j.contains("lodTree"))
                    meta.lod_tree = j.at("lodTree").get<bool>();
                if (j.contains("chunkSize"))
                    meta.chunk_size = j.at("chunkSize").get<uint32_t>();
                meta.all_chunk_bytes = j.at("allChunkBytes").get<uint64_t>();

                for (const auto& chunk_json : j.at("chunks")) {
                    meta.chunks.push_back(RadChunkRange::from_json(chunk_json));
                }

                if (j.contains("splatEncoding")) {
                    meta.splat_encoding = j.at("splatEncoding");
                }
                if (j.contains("shCodeCount")) {
                    meta.sh_code_count = j.at("shCodeCount").get<uint32_t>();
                }
                if (j.contains("comment")) {
                    meta.comment = j.at("comment").get<std::string>();
                }
                return meta;
            }
        };

        // ============================================================================
        // Property Encoding/Decoding
        // ============================================================================

        struct EncodedProperty {
            std::vector<uint8_t> data;
            std::string encoding;
            std::string compression;
            std::optional<float> min_val;
            std::optional<float> max_val;
            std::optional<float> base;
            std::optional<float> scale;
        };

        class PropertyEncoder {
        public:
            static EncodedProperty encode_center(const float* data, size_t dims, size_t count, RadCenterEncoding encoding) {
                EncodedProperty result;

                switch (encoding) {
                case RadCenterEncoding::F32:
                    result.data = encode_f32(data, dims, count);
                    result.encoding = "f32";
                    result.compression = "none";
                    break;

                case RadCenterEncoding::F32LeBytes:
                    result.data = encode_f32_lebytes(data, dims, count);
                    result.encoding = "f32_lebytes";
                    result.compression = "none";
                    break;

                case RadCenterEncoding::F16:
                    result.data = encode_f16(data, dims, count);
                    result.encoding = "f16";
                    result.compression = "none";
                    break;

                case RadCenterEncoding::F16LeBytes:
                    result.data = encode_f16_lebytes(data, dims, count);
                    result.encoding = "f16_lebytes";
                    result.compression = "none";
                    break;

                default:
                    // Match reference default behavior.
                    result.data = encode_f32_lebytes(data, dims, count);
                    result.encoding = "f32_lebytes";
                    result.compression = "none";
                    break;
                }

                return result;
            }

            static EncodedProperty encode_alpha(const float* data, size_t count, RadAlphaEncoding encoding, bool lod_tree) {
                EncodedProperty result;
                const float max_encoded_alpha = lod_tree ? 2.0f : 1.0f;

                switch (encoding) {
                case RadAlphaEncoding::F32:
                    result.data = encode_f32(data, 1, count);
                    result.encoding = "f32";
                    result.compression = "none";
                    break;

                case RadAlphaEncoding::F16:
                    result.data = encode_f16(data, 1, count);
                    result.encoding = "f16";
                    result.compression = "none";
                    break;

                case RadAlphaEncoding::R8: {
                    auto r8_result = encode_r8(data, 1, count, 0.0f, max_encoded_alpha);
                    result.data = std::move(r8_result.data);
                    result.min_val = r8_result.min_val;
                    result.max_val = r8_result.max_val;
                }
                    result.encoding = "r8";
                    result.compression = "none";
                    break;

                default: {
                    float max_alpha = 0.0f;
                    for (size_t i = 0; i < count; ++i)
                        max_alpha = std::max(max_alpha, data[i]);
                    if (max_alpha > 1.0f) {
                        result.data = encode_f16(data, 1, count);
                        result.encoding = "f16";
                    } else {
                        auto r8_result = encode_r8(data, 1, count, 0.0f, max_encoded_alpha);
                        result.data = std::move(r8_result.data);
                        result.min_val = r8_result.min_val;
                        result.max_val = r8_result.max_val;
                        result.encoding = "r8";
                    }
                }
                    result.compression = "none";
                    break;
                }

                return result;
            }

            static EncodedProperty encode_rgb(const float* data, size_t dims, size_t count, RadRgbEncoding encoding) {
                EncodedProperty result;

                switch (encoding) {
                case RadRgbEncoding::F32:
                    result.data = encode_f32(data, dims, count);
                    result.encoding = "f32";
                    result.compression = "none";
                    break;

                case RadRgbEncoding::F16:
                    result.data = encode_f16(data, dims, count);
                    result.encoding = "f16";
                    result.compression = "none";
                    break;

                case RadRgbEncoding::R8: {
                    auto r8_result = encode_r8(data, dims, count);
                    result.data = std::move(r8_result.data);
                    result.min_val = r8_result.min_val;
                    result.max_val = r8_result.max_val;
                }
                    result.encoding = "r8";
                    result.compression = "none";
                    break;

                case RadRgbEncoding::R8Delta: {
                    auto r8d_result = encode_r8_delta(data, dims, count);
                    result.data = std::move(r8d_result.data);
                    result.min_val = r8d_result.min_val;
                    result.max_val = r8d_result.max_val;
                }
                    result.encoding = "r8_delta";
                    result.compression = "none";
                    break;

                default: {
                    auto r8d_result = encode_r8_delta(data, dims, count);
                    result.data = std::move(r8d_result.data);
                    result.min_val = r8d_result.min_val;
                    result.max_val = r8d_result.max_val;
                }
                    result.encoding = "r8_delta";
                    result.compression = "none";
                    break;
                }

                return result;
            }

            static EncodedProperty encode_scales(const float* data, size_t dims, size_t count, RadScalesEncoding encoding) {
                EncodedProperty result;

                switch (encoding) {
                case RadScalesEncoding::F32:
                    result.data = encode_f32(data, dims, count);
                    result.encoding = "f32";
                    result.compression = "none";
                    break;

                case RadScalesEncoding::Ln0R8: {
                    auto ln_result = encode_ln_0r8(data, dims, count);
                    result.data = std::move(ln_result.data);
                    result.min_val = ln_result.min_val;
                    result.max_val = ln_result.max_val;
                }
                    result.encoding = "ln_0r8";
                    result.compression = "none";
                    break;

                case RadScalesEncoding::LnF16:
                    result.data = encode_ln_f16(data, dims, count);
                    result.encoding = "ln_f16";
                    result.compression = "none";
                    break;

                default:
                    result.data = encode_ln_f16(data, dims, count);
                    result.encoding = "ln_f16";
                    result.compression = "none";
                    break;
                }

                return result;
            }

            static EncodedProperty encode_orientation(const float* data, size_t count, RadOrientationEncoding encoding) {
                EncodedProperty result;
                std::vector<float> xyz(count * 3);
                for (size_t i = 0; i < count; ++i) {
                    xyz[i * 3 + 0] = data[i * 4 + 0];
                    xyz[i * 3 + 1] = data[i * 4 + 1];
                    xyz[i * 3 + 2] = data[i * 4 + 2];
                }

                switch (encoding) {
                case RadOrientationEncoding::F32:
                    result.data = encode_f32(xyz.data(), 3, count);
                    result.encoding = "f32";
                    result.compression = "none";
                    break;

                case RadOrientationEncoding::F16:
                    result.data = encode_f16(xyz.data(), 3, count);
                    result.encoding = "f16";
                    result.compression = "none";
                    break;

                case RadOrientationEncoding::Oct88R8:
                    result.data = encode_quat_oct88r8(data, count);
                    result.encoding = "oct88r8";
                    result.compression = "none";
                    break;

                default:
                    // Auto-detect: use oct88r8 for compact storage
                    result.data = encode_quat_oct88r8(data, count);
                    result.encoding = "oct88r8";
                    result.compression = "none";
                    break;
                }

                return result;
            }

            static EncodedProperty encode_sh(const float* data, size_t dims, size_t count, RadShEncoding encoding) {
                EncodedProperty result;

                switch (encoding) {
                case RadShEncoding::F32:
                    result.data = encode_f32(data, dims, count);
                    result.encoding = "f32";
                    result.compression = "none";
                    break;

                case RadShEncoding::F16:
                    result.data = encode_f16(data, dims, count);
                    result.encoding = "f16";
                    result.compression = "none";
                    break;

                case RadShEncoding::S8: {
                    auto s8_result = encode_s8(data, dims, count);
                    result.data.assign(reinterpret_cast<const uint8_t*>(s8_result.data.data()),
                                       reinterpret_cast<const uint8_t*>(s8_result.data.data() + s8_result.data.size()));
                    result.min_val = -s8_result.max_abs;
                    result.max_val = s8_result.max_abs;
                }
                    result.encoding = "s8";
                    result.compression = "none";
                    break;

                case RadShEncoding::S8Delta: {
                    auto s8d_result = encode_s8_delta(data, dims, count);
                    result.data.assign(reinterpret_cast<const uint8_t*>(s8d_result.data.data()),
                                       reinterpret_cast<const uint8_t*>(s8d_result.data.data() + s8d_result.data.size()));
                    result.min_val = -s8d_result.max_abs;
                    result.max_val = s8d_result.max_abs;
                }
                    result.encoding = "s8_delta";
                    result.compression = "none";
                    break;

                default: {
                    auto s8_result = encode_s8(data, dims, count);
                    result.data.assign(reinterpret_cast<const uint8_t*>(s8_result.data.data()),
                                       reinterpret_cast<const uint8_t*>(s8_result.data.data() + s8_result.data.size()));
                    result.min_val = -s8_result.max_abs;
                    result.max_val = s8_result.max_abs;
                }
                    result.encoding = "s8";
                    result.compression = "none";
                    break;
                }

                return result;
            }
        };

        class PropertyDecoder {
        public:
            static void decode_center(const uint8_t* data, float* output, size_t dims, size_t count,
                                      const std::string& encoding) {
                if (encoding == "f32") {
                    decode_f32(data, output, dims, count);
                } else if (encoding == "f32_lebytes") {
                    decode_f32_lebytes(data, output, dims, count);
                } else if (encoding == "f16") {
                    decode_f16(data, output, dims, count);
                } else if (encoding == "f16_lebytes") {
                    decode_f16_lebytes(data, output, dims, count);
                } else {
                    throw std::runtime_error("Unknown center encoding: " + encoding);
                }
            }

            static void decode_alpha(const uint8_t* data, float* output, size_t count,
                                     const std::string& encoding,
                                     float min_val, float max_val) {
                if (encoding == "f32") {
                    decode_f32(data, output, 1, count);
                } else if (encoding == "f16") {
                    decode_f16(data, output, 1, count);
                } else if (encoding == "r8") {
                    decode_r8(data, output, 1, count, min_val, max_val);
                } else {
                    throw std::runtime_error("Unknown alpha encoding: " + encoding);
                }
            }

            static void decode_rgb(const uint8_t* data, float* output, size_t dims, size_t count,
                                   const std::string& encoding,
                                   float min_val, float max_val,
                                   float, float) {
                if (encoding == "f32") {
                    decode_f32(data, output, dims, count);
                } else if (encoding == "f16") {
                    decode_f16(data, output, dims, count);
                } else if (encoding == "r8") {
                    decode_r8(data, output, dims, count, min_val, max_val);
                } else if (encoding == "r8_delta") {
                    decode_r8_delta(data, output, dims, count, min_val, max_val);
                } else {
                    throw std::runtime_error("Unknown RGB encoding: " + encoding);
                }
            }

            static void decode_scales(const uint8_t* data, float* output, size_t dims, size_t count,
                                      const std::string& encoding,
                                      float min_val, float max_val) {
                if (encoding == "f32") {
                    decode_f32(data, output, dims, count);
                } else if (encoding == "ln_0r8") {
                    decode_ln_0r8(data, output, dims, count, min_val, max_val);
                } else if (encoding == "ln_f16") {
                    decode_ln_f16(data, output, dims, count);
                } else {
                    throw std::runtime_error("Unknown scales encoding: " + encoding);
                }
            }

            static void decode_orientation(const uint8_t* data, float* output, size_t count,
                                           const std::string& encoding) {
                if (encoding == "f32") {
                    std::vector<float> xyz(count * 3);
                    decode_f32(data, xyz.data(), 3, count);
                    for (size_t i = 0; i < count; ++i) {
                        const float x = xyz[i * 3 + 0];
                        const float y = xyz[i * 3 + 1];
                        const float z = xyz[i * 3 + 2];
                        const float w = std::sqrt(std::max(0.0f, 1.0f - x * x - y * y - z * z));
                        output[i * 4 + 0] = x;
                        output[i * 4 + 1] = y;
                        output[i * 4 + 2] = z;
                        output[i * 4 + 3] = w;
                    }
                } else if (encoding == "f16") {
                    std::vector<float> xyz(count * 3);
                    decode_f16(data, xyz.data(), 3, count);
                    for (size_t i = 0; i < count; ++i) {
                        const float x = xyz[i * 3 + 0];
                        const float y = xyz[i * 3 + 1];
                        const float z = xyz[i * 3 + 2];
                        const float w = std::sqrt(std::max(0.0f, 1.0f - x * x - y * y - z * z));
                        output[i * 4 + 0] = x;
                        output[i * 4 + 1] = y;
                        output[i * 4 + 2] = z;
                        output[i * 4 + 3] = w;
                    }
                } else if (encoding == "oct88r8") {
                    decode_quat_oct88r8(data, output, count);
                } else {
                    throw std::runtime_error("Unknown orientation encoding: " + encoding);
                }
            }

            static void decode_sh(const uint8_t* data, float* output, size_t dims, size_t count,
                                  const std::string& encoding,
                                  float min_val, float max_val,
                                  float, float scale) {
                const float sh_max = std::max({std::abs(min_val), std::abs(max_val), std::abs(scale), 1e-6f});
                if (encoding == "f32") {
                    decode_f32(data, output, dims, count);
                } else if (encoding == "f16") {
                    decode_f16(data, output, dims, count);
                } else if (encoding == "s8") {
                    decode_s8(reinterpret_cast<const int8_t*>(data), output, dims, count, sh_max);
                } else if (encoding == "s8_delta") {
                    decode_s8_delta(reinterpret_cast<const int8_t*>(data), output, dims, count, sh_max);
                } else {
                    throw std::runtime_error("Unknown SH encoding: " + encoding);
                }
            }
        };

        // Decode one chunk's properties into caller-provided buffers holding
        // display-space values. Offsets are absolute positions in `data`;
        // `chunk_origin` anchors legacy chunk-relative property offsets.
        std::optional<std::string> decode_chunk_properties(
            const uint8_t* data,
            const RadChunkMeta& chunk,
            const size_t chunk_origin,
            const size_t payload_start,
            const bool has_payload_prefix,
            const size_t chunk_end,
            const int sh_coeffs,
            float* const means,
            float* const opacity,
            float* const sh0,
            float* const scales,
            float* const rotation,
            float* const shN,
            uint16_t* const child_count,
            uint32_t* const child_start) {
            const std::size_t chunk_count = static_cast<std::size_t>(chunk.count);
            std::vector<float> comp_data(chunk_count);

            try {
                for (const auto& prop : chunk.properties) {
                    const std::size_t prop_offset = static_cast<std::size_t>(prop.offset);
                    const std::size_t prop_bytes = static_cast<std::size_t>(prop.bytes);
                    const std::size_t absolute_offset =
                        has_payload_prefix ? (payload_start + prop_offset) : (chunk_origin + prop_offset);
                    if (absolute_offset + prop_bytes > chunk_end) {
                        return "RAD chunk property data exceeds file bounds";
                    }

                    std::vector<uint8_t> prop_data;
                    if (prop.compression.has_value() &&
                        (prop.compression.value() == "gz" || prop.compression.value() == "gzip")) {
                        const size_t decoded_bytes = rad_property_decoded_bytes(prop.property, prop.encoding, chunk_count);
                        prop_data = rad_decompress_sized(&data[absolute_offset], prop_bytes, decoded_bytes);
                        if (prop_data.empty()) {
                            return "Failed to decompress RAD chunk property: " + prop.property;
                        }
                    } else {
                        prop_data.assign(&data[absolute_offset], &data[absolute_offset + prop_bytes]);
                    }

                    if (prop.property == PROP_CENTER) {
                        PropertyDecoder::decode_center(prop_data.data(), means, 3, chunk_count, prop.encoding);
                    } else if (prop.property.find(PROP_CENTER) == 0 && prop.property != PROP_CENTER) {
                        const int comp = prop.property.back() - '0';
                        PropertyDecoder::decode_center(prop_data.data(), comp_data.data(), 1, chunk_count, prop.encoding);
                        for (std::size_t i = 0; i < chunk_count; ++i) {
                            means[i * 3u + static_cast<std::size_t>(comp)] = comp_data[i];
                        }
                    } else if (prop.property == PROP_ALPHA) {
                        PropertyDecoder::decode_alpha(prop_data.data(),
                                                      opacity,
                                                      chunk_count,
                                                      prop.encoding,
                                                      prop.min_val.value_or(0.0f),
                                                      prop.max_val.value_or(1.0f));
                    } else if (prop.property == PROP_RGB) {
                        PropertyDecoder::decode_rgb(prop_data.data(),
                                                    sh0,
                                                    3,
                                                    chunk_count,
                                                    prop.encoding,
                                                    prop.min_val.value_or(0.0f),
                                                    prop.max_val.value_or(1.0f),
                                                    prop.base.value_or(0.0f),
                                                    prop.scale.value_or(1.0f));
                    } else if (prop.property.find(PROP_RGB) == 0 && prop.property != PROP_RGB) {
                        const int comp = prop.property.back() - '0';
                        PropertyDecoder::decode_rgb(prop_data.data(),
                                                    comp_data.data(),
                                                    1,
                                                    chunk_count,
                                                    prop.encoding,
                                                    prop.min_val.value_or(0.0f),
                                                    prop.max_val.value_or(1.0f),
                                                    prop.base.value_or(0.0f),
                                                    prop.scale.value_or(1.0f));
                        for (std::size_t i = 0; i < chunk_count; ++i) {
                            sh0[i * 3u + static_cast<std::size_t>(comp)] = comp_data[i];
                        }
                    } else if (prop.property == PROP_SCALES) {
                        PropertyDecoder::decode_scales(prop_data.data(),
                                                       scales,
                                                       3,
                                                       chunk_count,
                                                       prop.encoding,
                                                       prop.min_val.value_or(0.0f),
                                                       prop.max_val.value_or(prop.scale.value_or(1.0f)));
                    } else if (prop.property.find(PROP_SCALES) == 0 && prop.property != PROP_SCALES) {
                        const int comp = prop.property.back() - '0';
                        PropertyDecoder::decode_scales(prop_data.data(),
                                                       comp_data.data(),
                                                       1,
                                                       chunk_count,
                                                       prop.encoding,
                                                       prop.min_val.value_or(0.0f),
                                                       prop.max_val.value_or(prop.scale.value_or(1.0f)));
                        for (std::size_t i = 0; i < chunk_count; ++i) {
                            scales[i * 3u + static_cast<std::size_t>(comp)] = comp_data[i];
                        }
                    } else if (prop.property == PROP_ORIENTATION) {
                        std::vector<float> quat_data(chunk_count * 4u);
                        PropertyDecoder::decode_orientation(prop_data.data(), quat_data.data(), chunk_count, prop.encoding);
                        for (std::size_t i = 0; i < chunk_count; ++i) {
                            rotation[i * 4u + 0u] = quat_data[i * 4u + 3u];
                            rotation[i * 4u + 1u] = quat_data[i * 4u + 0u];
                            rotation[i * 4u + 2u] = quat_data[i * 4u + 1u];
                            rotation[i * 4u + 3u] = quat_data[i * 4u + 2u];
                        }
                    } else if ((prop.property == PROP_SH1 || prop.property == PROP_SH2 || prop.property == PROP_SH3) &&
                               sh_coeffs > 0 && shN != nullptr) {
                        int coeff_start = 0;
                        int coeff_count = 0;
                        if (prop.property == PROP_SH1) {
                            coeff_start = 0;
                            coeff_count = 3;
                        } else if (prop.property == PROP_SH2) {
                            coeff_start = 3;
                            coeff_count = 5;
                        } else {
                            coeff_start = 8;
                            coeff_count = 7;
                        }

                        const std::size_t dims = static_cast<std::size_t>(coeff_count) * 3u;
                        std::vector<float> sh_block(chunk_count * dims, 0.0f);
                        PropertyDecoder::decode_sh(prop_data.data(),
                                                   sh_block.data(),
                                                   dims,
                                                   chunk_count,
                                                   prop.encoding,
                                                   prop.min_val.value_or(0.0f),
                                                   prop.max_val.value_or(1.0f),
                                                   prop.base.value_or(0.0f),
                                                   prop.scale.value_or(1.0f));

                        for (std::size_t i = 0; i < chunk_count; ++i) {
                            for (int c = 0; c < coeff_count; ++c) {
                                const int coeff = coeff_start + c;
                                if (coeff >= sh_coeffs) {
                                    continue;
                                }
                                for (int ch = 0; ch < 3; ++ch) {
                                    shN[i * static_cast<std::size_t>(sh_coeffs) * 3u +
                                        static_cast<std::size_t>(coeff) * 3u +
                                        static_cast<std::size_t>(ch)] =
                                        sh_block[i * dims + static_cast<std::size_t>(c) * 3u +
                                                 static_cast<std::size_t>(ch)];
                                }
                            }
                        }
                    } else if (prop.property.find("sh") == 0 && sh_coeffs > 0 && shN != nullptr) {
                        const std::size_t first_underscore = prop.property.find('_');
                        const std::size_t second_underscore = prop.property.find('_', first_underscore + 1);
                        if (first_underscore != std::string::npos && second_underscore != std::string::npos) {
                            const int coeff = std::stoi(prop.property.substr(first_underscore + 1,
                                                                             second_underscore - first_underscore - 1));
                            const int ch = prop.property.back() - '0';
                            if (coeff >= 0 && coeff < sh_coeffs && ch >= 0 && ch < 3) {
                                PropertyDecoder::decode_sh(prop_data.data(),
                                                           comp_data.data(),
                                                           1,
                                                           chunk_count,
                                                           prop.encoding,
                                                           prop.min_val.value_or(0.0f),
                                                           prop.max_val.value_or(1.0f),
                                                           prop.base.value_or(0.0f),
                                                           prop.scale.value_or(1.0f));
                                for (std::size_t i = 0; i < chunk_count; ++i) {
                                    shN[i * static_cast<std::size_t>(sh_coeffs) * 3u +
                                        static_cast<std::size_t>(coeff) * 3u +
                                        static_cast<std::size_t>(ch)] = comp_data[i];
                                }
                            }
                        }
                    } else if (prop.property == PROP_CHILD_COUNT && child_count != nullptr) {
                        if (prop_data.size() >= chunk_count * 2u) {
                            for (std::size_t i = 0; i < chunk_count; ++i) {
                                child_count[i] = decode_u16(&prop_data[i * 2u]);
                            }
                        }
                    } else if (prop.property == PROP_CHILD_START && child_start != nullptr) {
                        if (prop_data.size() >= chunk_count * 4u) {
                            for (std::size_t i = 0; i < chunk_count; ++i) {
                                child_start[i] = decode_u32(&prop_data[i * 4u]);
                            }
                        }
                    }
                }
            } catch (const std::exception& e) {
                return std::string("Failed to decode RAD chunk: ") + e.what();
            }
            return std::nullopt;
        }

        std::expected<RadDecodedChunk, std::string> decode_rad_chunk_buffer(
            const std::vector<uint8_t>& data,
            int fallback_max_sh,
            const bool has_lod_tree,
            bool lod_opacity_encoded) {
            if (data.size() < 8) {
                return std::unexpected("RAD chunk too small");
            }

            std::size_t offset = 0;
            const uint32_t chunk_magic = decode_u32(&data[offset]);
            if (chunk_magic != RAD_CHUNK_MAGIC) {
                return std::unexpected("Invalid RAD chunk magic");
            }

            const uint32_t chunk_meta_size = decode_u32(&data[offset + 4]);
            const std::size_t chunk_meta_padded = pad8(chunk_meta_size);
            if (offset + 8 + chunk_meta_padded + 8 > data.size()) {
                return std::unexpected("Unexpected end of RAD chunk metadata");
            }

            RadChunkMeta chunk;
            try {
                std::string chunk_json(reinterpret_cast<const char*>(&data[offset + 8]), chunk_meta_size);
                chunk = RadChunkMeta::from_json(nlohmann::json::parse(chunk_json));
            } catch (const std::exception& e) {
                return std::unexpected(std::string("Failed to parse RAD chunk metadata: ") + e.what());
            }

            const std::size_t payload_size_offset = offset + 8 + chunk_meta_padded;
            bool has_payload_prefix = false;
            std::size_t payload_start = payload_size_offset;
            std::size_t chunk_end = 0;
            if (payload_size_offset + 8 <= data.size()) {
                const uint64_t payload_bytes = decode_u64(&data[payload_size_offset]);
                payload_start = payload_size_offset + 8;
                chunk_end = payload_start + static_cast<std::size_t>(payload_bytes);
                has_payload_prefix = (chunk_end <= data.size()) && (chunk.payload_bytes == payload_bytes);
            }

            if (!has_payload_prefix) {
                payload_start = offset;
                chunk_end = offset + pad8(static_cast<std::size_t>(chunk.payload_bytes));
                if (chunk_end > data.size()) {
                    return std::unexpected("RAD chunk payload exceeds file bounds");
                }
            }

            if (chunk.splat_encoding.has_value()) {
                const auto& enc = chunk.splat_encoding.value();
                if (enc.is_object()) {
                    const auto it = enc.find("lodOpacity");
                    if (it != enc.end() && it->is_boolean()) {
                        lod_opacity_encoded = it->get<bool>();
                    }
                }
            }

            int max_sh = chunk.max_sh > 0 ? chunk.max_sh : fallback_max_sh;
            max_sh = std::clamp(max_sh, 0, 3);
            const int sh_coeffs = max_sh > 0 ? SH_COEFFS_FOR_DEGREE[max_sh] : 0;
            const std::size_t chunk_count = static_cast<std::size_t>(chunk.count);
            const bool decode_tree = has_lod_tree || chunk.lod_tree;

            std::vector<float> chunk_means(chunk_count * 3u);
            std::vector<float> chunk_opacity(chunk_count);
            std::vector<float> chunk_sh0(chunk_count * 3u);
            std::vector<float> chunk_scales(chunk_count * 3u);
            std::vector<float> chunk_rotation(chunk_count * 4u);
            std::vector<float> chunk_shN(chunk_count * static_cast<std::size_t>(sh_coeffs) * 3u, 0.0f);
            std::vector<uint16_t> chunk_child_count;
            std::vector<uint32_t> chunk_child_start;
            if (decode_tree) {
                chunk_child_count.resize(chunk_count);
                chunk_child_start.resize(chunk_count);
            }

            if (auto err = decode_chunk_properties(data.data(),
                                                   chunk,
                                                   offset,
                                                   payload_start,
                                                   has_payload_prefix,
                                                   chunk_end,
                                                   sh_coeffs,
                                                   chunk_means.data(),
                                                   chunk_opacity.data(),
                                                   chunk_sh0.data(),
                                                   chunk_scales.data(),
                                                   chunk_rotation.data(),
                                                   chunk_shN.empty() ? nullptr : chunk_shN.data(),
                                                   decode_tree ? chunk_child_count.data() : nullptr,
                                                   decode_tree ? chunk_child_start.data() : nullptr);
                err.has_value()) {
                return std::unexpected(std::move(*err));
            }

            for (float& v : chunk_sh0) {
                v = (v - 0.5f) / SH_C0;
            }
            if (!lod_opacity_encoded) {
                for (float& v : chunk_opacity) {
                    const float a = std::clamp(v, 1.0e-6f, 1.0f - 1.0e-6f);
                    v = std::log(a / (1.0f - a));
                }
            } else {
                for (float& v : chunk_opacity) {
                    v = std::max(v, 0.0f);
                }
            }
            for (float& v : chunk_scales) {
                v = std::log(std::max(v, 1.0e-8f));
            }

            return RadDecodedChunk{
                .base = chunk.base,
                .count = chunk.count,
                .max_sh_degree = max_sh,
                .sh_coeffs_rest = static_cast<std::uint32_t>(sh_coeffs),
                .lod_opacity_encoded = lod_opacity_encoded,
                .means = std::move(chunk_means),
                .opacity_raw = std::move(chunk_opacity),
                .sh0_raw = std::move(chunk_sh0),
                .scaling_raw = std::move(chunk_scales),
                .rotation_raw = std::move(chunk_rotation),
                .shN_canonical = std::move(chunk_shN),
                .child_count = std::move(chunk_child_count),
                .child_start = std::move(chunk_child_start),
            };
        }

        // ============================================================================
        // RAD Encoder
        // ============================================================================

        class RadEncoder {
        public:
            explicit RadEncoder(int compression_level = GZ_LEVEL,
                                bool flip_y = false,
                                ExportProgressCallback progress_callback = nullptr)
                : compression_level_(compression_level),
                  flip_y_(flip_y),
                  progress_callback_(std::move(progress_callback)) {}

            std::vector<uint8_t> encode(const SplatData& splat_data) {
                // 0.0: Preparing data
                if (!report_progress(0.0f, "Preparing data...")) {
                    throw std::runtime_error("CANCELLED");
                }

                std::optional<SplatData> visible_splat_data;
                std::optional<SplatData> lod_splat_data;
                const SplatData* export_source = &splat_data;

                const bool has_deleted = splat_data.has_deleted_mask() && splat_data.deleted().count_nonzero() > 0;

                if (has_deleted) {
                    const Tensor keep_mask = splat_data.deleted().logical_not();
                    auto extracted = lfs::core::extract_by_mask(splat_data, keep_mask);
                    if (extracted.size() > 0) {
                        visible_splat_data = std::move(extracted);
                        export_source = &visible_splat_data.value();
                    }
                }

                // Build LOD tree if the source doesn't have one.
                if (!export_source->lod_tree || !export_source->lod_tree->has_tree()) {
                    auto lod_progress = [&](float p, const std::string& stage) -> bool {
                        return report_progress(p * 0.1f, stage);
                    };
                    auto lod_result = lfs::core::build_bhatt_lod(*export_source, 1.25f, lod_progress);
                    if (lod_result && (*lod_result)->lod_tree && (*lod_result)->lod_tree->has_tree()) {
                        lod_splat_data = std::move(**lod_result);
                        export_source = &lod_splat_data.value();
                    }
                }

                // 0.1: Packing splat data
                if (!report_progress(0.1f, "Packing splat data...")) {
                    throw std::runtime_error("CANCELLED");
                }

                PackedSplatData packed = pack_splat_data(*export_source, flip_y_);

                // 0.2: Data packed
                if (!report_progress(0.2f, "Data packed")) {
                    throw std::runtime_error("CANCELLED");
                }

                // 0.3: Preparing chunks
                if (!report_progress(0.3f, "Preparing chunks...")) {
                    throw std::runtime_error("CANCELLED");
                }

                if (packed.count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
                    throw std::runtime_error("RAD export exceeds maximum supported splat count");
                }

                const uint32_t num_splats = static_cast<uint32_t>(packed.count);
                const int sh_degree = packed.sh_degree;
                const int sh_coeffs = packed.sh_coeffs;
                const bool lod_tree = packed.lod_tree;

                const uint32_t num_chunks = (num_splats + CHUNK_SIZE - 1) / CHUNK_SIZE;

                // Build metadata
                RadMeta meta;
                meta.count = num_splats;
                meta.max_sh = sh_degree;
                meta.lod_tree = lod_tree ? std::optional<bool>(true) : std::nullopt;
                meta.chunk_size = CHUNK_SIZE;
                if (lod_tree) {
                    meta.splat_encoding = nlohmann::json{{"lodOpacity", true}};
                }

                // Encode chunks in parallel. lfs_io already links TBB, so keep
                // RAD export on the same threading runtime as the rest of IO.
                std::vector<std::vector<uint8_t>> chunk_payloads(num_chunks);
                std::vector<RadChunkRange> chunk_ranges(num_chunks);
                std::atomic<uint32_t> completed_chunks{0};

                // Report initial progress
                if (!report_progress(0.5f, "Encoding chunks...")) {
                    throw std::runtime_error("CANCELLED");
                }

                tbb::parallel_for(
                    tbb::blocked_range<uint32_t>(0, num_chunks, 1),
                    [&](const tbb::blocked_range<uint32_t>& range) {
                        for (uint32_t chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
                            const uint32_t base = chunk_idx * CHUNK_SIZE;
                            const uint32_t count = std::min(CHUNK_SIZE, num_splats - base);

                            auto chunk_progress_cb = [&](float /*progress*/) -> bool {
                                const uint32_t completed = completed_chunks.load(std::memory_order_relaxed);
                                if (completed % 16 == 0) {
                                    const float overall = 0.5f + (0.4f * static_cast<float>(completed) / static_cast<float>(num_chunks));
                                    return report_progress(overall, "Encoding chunks...");
                                }
                                return true;
                            };

                            auto chunk_result = encode_chunk(
                                base, count, sh_degree, sh_coeffs,
                                packed.means,
                                packed.opacity,
                                packed.sh0,
                                packed.scales,
                                packed.rotation,
                                packed.shN,
                                lod_tree ? packed.child_count.data() : nullptr,
                                lod_tree ? packed.child_start.data() : nullptr,
                                lod_tree,
                                chunk_progress_cb);

                            chunk_ranges[chunk_idx].base = base;
                            chunk_ranges[chunk_idx].count = count;
                            chunk_ranges[chunk_idx].bytes = chunk_result.second.size();
                            chunk_payloads[chunk_idx] = std::move(chunk_result.second);

                            completed_chunks.fetch_add(1, std::memory_order_relaxed);
                        }
                    });

                // Build metadata in order (sequential - must preserve chunk order)
                uint64_t current_chunk_offset = 0;
                for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                    chunk_ranges[chunk_idx].offset = current_chunk_offset;
                    meta.chunks.push_back(chunk_ranges[chunk_idx]);
                    current_chunk_offset += chunk_ranges[chunk_idx].bytes;
                }

                // Calculate total chunk bytes
                for (const auto& payload : chunk_payloads) {
                    meta.all_chunk_bytes += payload.size();
                }

                // Serialize metadata to JSON
                std::string meta_json = meta.to_json().dump();

                const size_t meta_size = meta_json.size();
                const size_t meta_padded_size = pad8(meta_size);

                // Build header: RAD_MAGIC (4 bytes) + metadata_length (4 bytes) = 8 bytes total
                std::vector<uint8_t> header(8);
                encode_u32(&header[0], RAD_MAGIC);
                encode_u32(&header[4], static_cast<uint32_t>(meta_size));

                // 0.9: Assembling file data
                if (!report_progress(0.9f, "Assembling RAD data...")) {
                    throw std::runtime_error("CANCELLED");
                }

                // Combine all data
                std::vector<uint8_t> result;
                result.reserve(header.size() + meta_padded_size + meta.all_chunk_bytes);

                result.insert(result.end(), header.begin(), header.end());
                result.insert(result.end(), meta_json.begin(), meta_json.end());
                if (meta_padded_size > meta_size) {
                    result.insert(result.end(), meta_padded_size - meta_size, 0);
                }

                for (const auto& payload : chunk_payloads) {
                    result.insert(result.end(), payload.begin(), payload.end());
                }

                // 1.0: Encoding complete
                if (!report_progress(1.0f, "RAD data prepared")) {
                    throw std::runtime_error("CANCELLED");
                }

                return result;
            }

        private:
            // Holds CPU-resident tensors (or transformed copies where the RAD
            // domain differs) and exposes raw pointers for chunk encoding.
            struct PackedSplatData {
                size_t count = 0;
                int sh_degree = 0;
                int sh_coeffs = 0;
                bool lod_tree = false;
                Tensor means_storage;
                Tensor opacity_storage;
                Tensor scales_storage;
                Tensor shN_storage;
                std::vector<float> means_flipped;
                std::vector<float> sh0_display;
                std::vector<float> rotation_normalized;
                const float* means = nullptr;
                const float* opacity = nullptr;
                const float* sh0 = nullptr;
                const float* scales = nullptr;
                const float* rotation = nullptr;
                const float* shN = nullptr;
                std::vector<uint16_t> child_count;
                std::vector<uint32_t> child_start;
            };

            static PackedSplatData pack_splat_data(const SplatData& splat_data, bool flip_y = false) {
                PackedSplatData packed;
                packed.count = static_cast<size_t>(splat_data.size());
                packed.sh_degree = std::clamp(splat_data.get_max_sh_degree(), 0, 3);
                packed.sh_coeffs = packed.sh_degree > 0 ? SH_COEFFS_FOR_DEGREE[packed.sh_degree] : 0;
                if (packed.count == 0) {
                    return packed;
                }

                auto cpu_contiguous = [](Tensor tensor) {
                    return tensor.contiguous().to(Device::CPU);
                };

                // Spark RAD stores render-space values, not optimizer-domain tensors.
                packed.means_storage = cpu_contiguous(splat_data.get_means());
                packed.means = packed.means_storage.ptr<float>();
                if (flip_y) {
                    const float* const src = packed.means;
                    packed.means_flipped.resize(packed.count * 3);
                    tbb::parallel_for(
                        tbb::blocked_range<size_t>(0, packed.count),
                        [&](const tbb::blocked_range<size_t>& range) {
                            for (size_t i = range.begin(); i != range.end(); ++i) {
                                packed.means_flipped[i * 3 + 0] = src[i * 3 + 0];
                                packed.means_flipped[i * 3 + 1] = -src[i * 3 + 1];
                                packed.means_flipped[i * 3 + 2] = src[i * 3 + 2];
                            }
                        });
                    packed.means = packed.means_flipped.data();
                }

                if (splat_data.lod_tree && splat_data.lod_tree->lod_opacity_encoded) {
                    // For LOD-encoded opacity, read raw values directly (display-space, can exceed 1.0)
                    packed.opacity_storage = cpu_contiguous(splat_data.opacity_raw());
                } else {
                    packed.opacity_storage = cpu_contiguous(splat_data.get_opacity());
                }
                packed.opacity = packed.opacity_storage.ptr<float>();

                const Tensor sh0_cpu = cpu_contiguous(splat_data.sh0_raw());
                const float* const sh0_src = sh0_cpu.ptr<float>();
                packed.sh0_display.resize(packed.count * 3);
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, packed.sh0_display.size()),
                    [&](const tbb::blocked_range<size_t>& range) {
                        for (size_t i = range.begin(); i != range.end(); ++i) {
                            packed.sh0_display[i] = 0.5f + SH_C0 * sh0_src[i];
                        }
                    });
                packed.sh0 = packed.sh0_display.data();

                packed.scales_storage = cpu_contiguous(splat_data.get_scaling());
                packed.scales = packed.scales_storage.ptr<float>();

                // Normalize quaternions directly: the tensor-op chain
                // (square/sum/sqrt/div) materializes four temporaries and
                // dominates pack time on CPU.
                const Tensor rotation_cpu = cpu_contiguous(splat_data.rotation_raw());
                const float* const rot_src = rotation_cpu.ptr<float>();
                packed.rotation_normalized.resize(packed.count * 4);
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, packed.count),
                    [&](const tbb::blocked_range<size_t>& range) {
                        for (size_t i = range.begin(); i != range.end(); ++i) {
                            const float x = rot_src[i * 4 + 0];
                            const float y = rot_src[i * 4 + 1];
                            const float z = rot_src[i * 4 + 2];
                            const float w = rot_src[i * 4 + 3];
                            // Match the tensor-op reference bit-for-bit: float
                            // squares, double-accumulated sum, float sqrt.
                            double sum_squared = 0.0;
                            sum_squared += x * x;
                            sum_squared += y * y;
                            sum_squared += z * z;
                            sum_squared += w * w;
                            const float norm = std::max(std::sqrt(static_cast<float>(sum_squared)), 1e-12f);
                            packed.rotation_normalized[i * 4 + 0] = x / norm;
                            packed.rotation_normalized[i * 4 + 1] = y / norm;
                            packed.rotation_normalized[i * 4 + 2] = z / norm;
                            packed.rotation_normalized[i * 4 + 3] = w / norm;
                        }
                    });
                packed.rotation = packed.rotation_normalized.data();
                if (packed.sh_coeffs > 0) {
                    // shN is stored swizzled; unpack on CPU to avoid a canonical CUDA copy.
                    packed.shN_storage = splat_data.shN_canonical_cpu();
                    packed.shN = packed.shN_storage.ptr<float>();
                }
                if (splat_data.lod_tree && splat_data.lod_tree->has_tree()) {
                    packed.lod_tree = true;
                    packed.child_count = splat_data.lod_tree->child_count;
                    packed.child_start = splat_data.lod_tree->child_start;
                }

                return packed;
            }

            std::pair<RadChunkMeta, std::vector<uint8_t>> encode_chunk(
                uint32_t base, uint32_t count, int sh_degree, int sh_coeffs,
                const float* means_ptr,
                const float* opacity_ptr,
                const float* sh0_ptr,
                const float* scales_ptr,
                const float* rotation_ptr,
                const float* shN_ptr,
                const uint16_t* child_count_ptr,
                const uint32_t* child_start_ptr,
                bool lod_tree,
                const std::function<bool(float)>& progress_callback = nullptr) {

                RadChunkMeta chunk_meta;
                chunk_meta.version = 1;
                chunk_meta.base = base;
                chunk_meta.count = count;
                chunk_meta.max_sh = sh_degree;
                chunk_meta.lod_tree = lod_tree;
                if (lod_tree) {
                    chunk_meta.splat_encoding = nlohmann::json{{"lodOpacity", true}};
                }

                std::vector<EncodedProperty> encoded_props;
                encoded_props.reserve(lod_tree ? 12 : 10);

                // Thread-local buffers for temporary data to avoid allocation contention
                thread_local std::vector<float> tl_sh_data;

                // Encode center (3 components together as single property)
                {
                    // Encode all 3 components together as "center" property
                    auto encoded = PropertyEncoder::encode_center(means_ptr + static_cast<size_t>(base) * 3, 3, count, RadCenterEncoding::Auto);
                    auto compressed = rad_compress(encoded.data.data(), encoded.data.size(), compression_level_);

                    RadChunkProperty prop;
                    prop.property = PROP_CENTER;
                    prop.encoding = encoded.encoding;
                    prop.compression = "gz";
                    prop.bytes = compressed.size();

                    encoded_props.push_back({std::move(compressed), encoded.encoding, "gz",
                                             encoded.min_val, encoded.max_val, encoded.base, encoded.scale});
                    chunk_meta.properties.push_back(prop);

                    // Report progress after encoding center: 0.1f
                    if (progress_callback && !progress_callback(0.1f)) {
                        throw std::runtime_error("CANCELLED");
                    }
                }

                // Encode alpha
                {
                    auto encoded = PropertyEncoder::encode_alpha(opacity_ptr + base, count, RadAlphaEncoding::Auto, lod_tree);
                    auto compressed = rad_compress(encoded.data.data(), encoded.data.size(), compression_level_);

                    RadChunkProperty prop;
                    prop.property = PROP_ALPHA;
                    prop.encoding = encoded.encoding;
                    prop.compression = "gz";
                    prop.bytes = compressed.size();
                    if (encoded.min_val.has_value())
                        prop.min_val = encoded.min_val.value();
                    if (encoded.max_val.has_value())
                        prop.max_val = encoded.max_val.value();

                    encoded_props.push_back({std::move(compressed), encoded.encoding, "gz",
                                             encoded.min_val, encoded.max_val, encoded.base, encoded.scale});
                    chunk_meta.properties.push_back(prop);

                    // Report progress after encoding alpha: 0.2f
                    if (progress_callback && !progress_callback(0.2f)) {
                        throw std::runtime_error("CANCELLED");
                    }
                }

                // Encode RGB (sh0) - all 3 components together as single property
                {
                    // Encode all 3 components together as "rgb" property
                    auto encoded = PropertyEncoder::encode_rgb(sh0_ptr + static_cast<size_t>(base) * 3, 3, count, RadRgbEncoding::Auto);
                    auto compressed = rad_compress(encoded.data.data(), encoded.data.size(), compression_level_);

                    RadChunkProperty prop;
                    prop.property = PROP_RGB;
                    prop.encoding = encoded.encoding;
                    prop.compression = "gz";
                    prop.bytes = compressed.size();
                    if (encoded.min_val.has_value())
                        prop.min_val = encoded.min_val.value();
                    if (encoded.max_val.has_value())
                        prop.max_val = encoded.max_val.value();
                    if (encoded.base.has_value())
                        prop.base = encoded.base.value();
                    if (encoded.scale.has_value())
                        prop.scale = encoded.scale.value();

                    encoded_props.push_back({std::move(compressed), encoded.encoding, "gz",
                                             encoded.min_val, encoded.max_val, encoded.base, encoded.scale});
                    chunk_meta.properties.push_back(prop);

                    // Report progress after encoding RGB: 0.4f
                    if (progress_callback && !progress_callback(0.4f)) {
                        throw std::runtime_error("CANCELLED");
                    }
                }

                // Encode scales - all 3 components together as single property
                {
                    // Encode all 3 components together as "scales" property
                    auto encoded = PropertyEncoder::encode_scales(scales_ptr + static_cast<size_t>(base) * 3, 3, count, RadScalesEncoding::Auto);
                    auto compressed = rad_compress(encoded.data.data(), encoded.data.size(), compression_level_);

                    RadChunkProperty prop;
                    prop.property = PROP_SCALES;
                    prop.encoding = encoded.encoding;
                    prop.compression = "gz";
                    prop.bytes = compressed.size();
                    if (encoded.min_val.has_value())
                        prop.min_val = encoded.min_val.value();
                    if (encoded.max_val.has_value())
                        prop.max_val = encoded.max_val.value();
                    if (encoded.scale.has_value())
                        prop.scale = encoded.scale.value();

                    encoded_props.push_back({std::move(compressed), encoded.encoding, "gz",
                                             encoded.min_val, encoded.max_val, encoded.base, encoded.scale});
                    chunk_meta.properties.push_back(prop);

                    // Report progress after encoding scales: 0.6f
                    if (progress_callback && !progress_callback(0.6f)) {
                        throw std::runtime_error("CANCELLED");
                    }
                }

                // Encode orientation
                {
                    EncodedProperty encoded;
                    encoded.data = encode_quat_oct88r8(rotation_ptr + static_cast<size_t>(base) * 4, count, true);
                    encoded.encoding = "oct88r8";
                    encoded.compression = "none";
                    auto compressed = rad_compress(encoded.data.data(), encoded.data.size(), compression_level_);

                    RadChunkProperty prop;
                    prop.property = PROP_ORIENTATION;
                    prop.encoding = encoded.encoding;
                    prop.compression = "gz";
                    prop.bytes = compressed.size();

                    encoded_props.push_back({std::move(compressed), encoded.encoding, "gz",
                                             encoded.min_val, encoded.max_val, encoded.base, encoded.scale});
                    chunk_meta.properties.push_back(prop);

                    // Report progress after encoding orientation: 0.8f
                    if (progress_callback && !progress_callback(0.8f)) {
                        throw std::runtime_error("CANCELLED");
                    }
                }

                // Encode SH if present
                if (sh_coeffs > 0 && shN_ptr != nullptr) {
                    auto encode_sh_band = [&](const char* prop_name, int coeff_start, int coeff_count) {
                        if (sh_coeffs < coeff_start + coeff_count) {
                            return;
                        }
                        const size_t dims = static_cast<size_t>(coeff_count) * 3;
                        tl_sh_data.resize(static_cast<size_t>(count) * dims);
                        for (uint32_t i = 0; i < count; ++i) {
                            for (int c = 0; c < coeff_count; ++c) {
                                for (int ch = 0; ch < 3; ++ch) {
                                    tl_sh_data[i * dims + c * 3 + ch] =
                                        shN_ptr[(base + i) * sh_coeffs * 3 + (coeff_start + c) * 3 + ch];
                                }
                            }
                        }

                        auto encoded = PropertyEncoder::encode_sh(tl_sh_data.data(), dims, count, RadShEncoding::Auto);
                        auto compressed = rad_compress(encoded.data.data(), encoded.data.size(), compression_level_);

                        RadChunkProperty prop;
                        prop.property = prop_name;
                        prop.encoding = encoded.encoding;
                        prop.compression = "gz";
                        prop.bytes = compressed.size();
                        if (encoded.min_val.has_value())
                            prop.min_val = encoded.min_val.value();
                        if (encoded.max_val.has_value())
                            prop.max_val = encoded.max_val.value();
                        if (encoded.base.has_value())
                            prop.base = encoded.base.value();
                        if (encoded.scale.has_value())
                            prop.scale = encoded.scale.value();

                        encoded_props.push_back({std::move(compressed), encoded.encoding, "gz",
                                                 encoded.min_val, encoded.max_val, encoded.base, encoded.scale});
                        chunk_meta.properties.push_back(prop);
                    };

                    encode_sh_band(PROP_SH1, 0, 3);
                    encode_sh_band(PROP_SH2, 3, 5);
                    encode_sh_band(PROP_SH3, 8, 7);

                    // Report progress after encoding SH: 0.9f
                    if (progress_callback && !progress_callback(0.9f)) {
                        throw std::runtime_error("CANCELLED");
                    }
                }

                if (lod_tree && child_count_ptr != nullptr && child_start_ptr != nullptr) {
                    // Encode child_count
                    std::vector<uint8_t> child_count_data(static_cast<size_t>(count) * 2);
                    for (uint32_t i = 0; i < count; ++i) {
                        encode_u16(child_count_data.data() + static_cast<size_t>(i) * 2, child_count_ptr[base + i]);
                    }
                    auto child_count_compressed = rad_compress(child_count_data.data(), child_count_data.size(), compression_level_);

                    RadChunkProperty count_prop;
                    count_prop.property = PROP_CHILD_COUNT;
                    count_prop.encoding = "u16";
                    count_prop.compression = "gz";
                    count_prop.bytes = child_count_compressed.size();

                    encoded_props.push_back({std::move(child_count_compressed), "u16", "gz",
                                             std::nullopt, std::nullopt, std::nullopt, std::nullopt});
                    chunk_meta.properties.push_back(count_prop);

                    // Encode child_start
                    std::vector<uint8_t> child_start_data(static_cast<size_t>(count) * 4);
                    for (uint32_t i = 0; i < count; ++i) {
                        encode_u32(child_start_data.data() + static_cast<size_t>(i) * 4, child_start_ptr[base + i]);
                    }
                    auto child_start_compressed = rad_compress(child_start_data.data(), child_start_data.size(), compression_level_);

                    RadChunkProperty start_prop;
                    start_prop.property = PROP_CHILD_START;
                    start_prop.encoding = "u32";
                    start_prop.compression = "gz";
                    start_prop.bytes = child_start_compressed.size();

                    encoded_props.push_back({std::move(child_start_compressed), "u32", "gz",
                                             std::nullopt, std::nullopt, std::nullopt, std::nullopt});
                    chunk_meta.properties.push_back(start_prop);

                    // Report progress after encoding LOD data: 0.95f
                    if (progress_callback && !progress_callback(0.95f)) {
                        throw std::runtime_error("CANCELLED");
                    }
                }

                std::vector<uint8_t> payload;
                // Property offsets are payload-relative (start at first property byte),
                // not chunk-relative. This matches Spark's RAD decoder semantics where
                // absolute_offset = payload_start + prop.offset.
                uint64_t payload_bytes = 0;
                for (size_t i = 0; i < encoded_props.size(); ++i) {
                    chunk_meta.properties[i].offset = payload_bytes;
                    size_t prop_size = encoded_props[i].data.size();
                    size_t padded_size = pad8(prop_size);
                    payload_bytes += padded_size;
                }

                chunk_meta.payload_bytes = payload_bytes;

                std::string chunk_json = chunk_meta.to_json().dump();
                const size_t chunk_json_size = chunk_json.size();
                const size_t chunk_json_padded = pad8(chunk_json_size);

                payload.reserve(8 + chunk_json_padded + 8 + static_cast<size_t>(payload_bytes));
                payload.resize(8);
                encode_u32(payload.data(), RAD_CHUNK_MAGIC);
                encode_u32(payload.data() + 4, static_cast<uint32_t>(chunk_json_size));

                payload.insert(payload.end(), chunk_json.begin(), chunk_json.end());
                if (chunk_json_padded > chunk_json_size) {
                    payload.insert(payload.end(), chunk_json_padded - chunk_json_size, 0);
                }

                uint8_t payload_bytes_buf[8];
                encode_u64(payload_bytes_buf, payload_bytes);
                payload.insert(payload.end(), payload_bytes_buf, payload_bytes_buf + 8);

                for (size_t i = 0; i < encoded_props.size(); ++i) {
                    size_t prop_size = encoded_props[i].data.size();
                    size_t padded_size = pad8(prop_size);

                    payload.insert(payload.end(), encoded_props[i].data.begin(), encoded_props[i].data.end());
                    if (padded_size > prop_size) {
                        payload.insert(payload.end(), padded_size - prop_size, 0);
                    }
                }

                return {chunk_meta, payload};
            }

            int compression_level_;
            bool flip_y_;
            ExportProgressCallback progress_callback_;

            bool report_progress(float progress, const std::string& stage) const {
                if (progress_callback_) {
                    return progress_callback_(progress, stage);
                }
                return true;
            }
        };

        // ============================================================================
        // RAD Decoder
        // ============================================================================

        class RadDecoder {
        public:
            std::expected<SplatData, std::string> decode(
                const std::vector<uint8_t>& data,
                const std::filesystem::path* source_path = nullptr) {
                if (data.size() < 8) {
                    return std::unexpected("RAD file too small");
                }

                // Read header: 8 bytes (magic + metadata length)
                uint32_t magic = decode_u32(&data[0]);
                if (magic != RAD_MAGIC) {
                    return std::unexpected("Invalid RAD magic number");
                }

                uint32_t meta_size = decode_u32(&data[4]);

                // Read and parse metadata
                if (8 + meta_size > data.size()) {
                    return std::unexpected("RAD metadata size exceeds file size");
                }

                std::string meta_json(reinterpret_cast<const char*>(&data[8]), meta_size);
                // Trim padding spaces
                size_t actual_size = meta_json.find_last_not_of(' ');
                if (actual_size != std::string::npos) {
                    meta_json.resize(actual_size + 1);
                }

                RadMeta meta;
                try {
                    meta = RadMeta::from_json(nlohmann::json::parse(meta_json));
                } catch (const std::exception& e) {
                    return std::unexpected(std::string("Failed to parse RAD metadata: ") + e.what());
                }

                // Decode chunks
                size_t offset = 8 + pad8(meta_size);

                const int max_sh = meta.max_sh.value_or(0);
                const int sh_coeffs = max_sh > 0 ? SH_COEFFS_FOR_DEGREE[max_sh] : 0;
                const bool has_lod_tree = meta.lod_tree.value_or(false);
                const size_t N = meta.count;

                bool lod_opacity_encoded = has_lod_tree;
                if (meta.splat_encoding.has_value()) {
                    const auto& enc = meta.splat_encoding.value();
                    if (enc.is_object()) {
                        auto it = enc.find("lodOpacity");
                        if (it != enc.end() && it->is_boolean()) {
                            lod_opacity_encoded = it->get<bool>();
                        }
                    }
                }

                struct ChunkSlice {
                    RadChunkMeta meta;
                    size_t origin = 0;
                    size_t payload_start = 0;
                    size_t chunk_end = 0;
                    bool has_payload_prefix = false;
                };

                std::vector<ChunkSlice> slices;
                slices.reserve(meta.chunks.size());
                std::vector<lfs::core::SplatLodTree::ChunkFileRange> chunk_file_ranges;
                if (has_lod_tree) {
                    chunk_file_ranges.reserve(meta.chunks.size());
                }

                uint64_t scanned_count = 0;
                for (size_t chunk_idx = 0; chunk_idx < meta.chunks.size(); ++chunk_idx) {
                    const size_t chunk_file_offset = offset;
                    if (offset + 8 > data.size()) {
                        return std::unexpected("Unexpected end of RAD file (chunk header)");
                    }

                    uint32_t chunk_magic = decode_u32(&data[offset]);
                    if (chunk_magic != RAD_CHUNK_MAGIC) {
                        return std::unexpected("Invalid RAD chunk magic");
                    }

                    uint32_t chunk_meta_size = decode_u32(&data[offset + 4]);
                    const size_t chunk_meta_padded = pad8(chunk_meta_size);

                    if (offset + 8 + chunk_meta_padded + 8 > data.size()) {
                        return std::unexpected("Unexpected end of RAD file (chunk metadata)");
                    }

                    std::string chunk_json(reinterpret_cast<const char*>(&data[offset + 8]), chunk_meta_size);

                    RadChunkMeta chunk;
                    try {
                        chunk = RadChunkMeta::from_json(nlohmann::json::parse(chunk_json));
                    } catch (const std::exception& e) {
                        return std::unexpected(std::string("Failed to parse chunk metadata: ") + e.what());
                    }

                    const size_t payload_size_offset = offset + 8 + chunk_meta_padded;
                    bool has_payload_prefix = false;
                    size_t payload_start = payload_size_offset;
                    size_t chunk_end = 0;
                    if (payload_size_offset + 8 <= data.size()) {
                        const uint64_t payload_bytes = decode_u64(&data[payload_size_offset]);
                        payload_start = payload_size_offset + 8;
                        chunk_end = payload_start + static_cast<size_t>(payload_bytes);
                        has_payload_prefix = (chunk_end <= data.size()) && (chunk.payload_bytes == payload_bytes);
                    }

                    // Legacy fallback: old C++ layout did not include payload_bytes after chunk metadata.
                    if (!has_payload_prefix) {
                        payload_start = offset;
                        chunk_end = offset + pad8(static_cast<size_t>(chunk.payload_bytes));
                        if (chunk_end > data.size()) {
                            return std::unexpected("Chunk payload exceeds file bounds");
                        }
                    }
                    if (has_lod_tree) {
                        chunk_file_ranges.push_back({
                            .file_offset = static_cast<uint64_t>(chunk_file_offset),
                            .file_bytes = static_cast<uint64_t>(chunk_end - chunk_file_offset),
                            .payload_offset = static_cast<uint64_t>(payload_start),
                            .payload_bytes = chunk.payload_bytes,
                            .base = chunk.base,
                            .count = chunk.count,
                        });
                    }

                    // Chunks must tile [0, N) in file order for the parallel
                    // slice writes below to be disjoint and complete.
                    if (chunk.base != scanned_count || chunk.base + chunk.count > N) {
                        return std::unexpected("RAD chunk base/count layout is inconsistent");
                    }
                    scanned_count += chunk.count;

                    slices.push_back({std::move(chunk), chunk_file_offset, payload_start, chunk_end, has_payload_prefix});
                    offset = chunk_end;
                }
                if (scanned_count != N) {
                    return std::unexpected("RAD chunk counts do not sum to splat count");
                }

                // Decode straight into the output tensors, one chunk per task.
                Tensor means_tensor = Tensor::empty({N, 3}, Device::CPU, lfs::core::DataType::Float32);
                Tensor opacity_tensor = Tensor::empty({N, 1}, Device::CPU, lfs::core::DataType::Float32);
                Tensor sh0_tensor = Tensor::empty({N, 1, 3}, Device::CPU, lfs::core::DataType::Float32);
                Tensor scales_tensor = Tensor::empty({N, 3}, Device::CPU, lfs::core::DataType::Float32);
                Tensor rotation_tensor = Tensor::empty({N, 4}, Device::CPU, lfs::core::DataType::Float32);
                Tensor shN_tensor;
                if (sh_coeffs > 0) {
                    shN_tensor = Tensor::empty({N, static_cast<size_t>(sh_coeffs), 3}, Device::CPU,
                                               lfs::core::DataType::Float32);
                }

                float* const all_means = means_tensor.ptr<float>();
                float* const all_opacity = opacity_tensor.ptr<float>();
                float* const all_sh0 = sh0_tensor.ptr<float>();
                float* const all_scales = scales_tensor.ptr<float>();
                float* const all_rotation = rotation_tensor.ptr<float>();
                float* const all_shN = sh_coeffs > 0 ? shN_tensor.ptr<float>() : nullptr;

                std::vector<uint16_t> all_child_count;
                std::vector<uint32_t> all_child_start;
                auto tree = has_lod_tree ? std::make_unique<lfs::core::SplatLodTree>() : nullptr;
                if (has_lod_tree) {
                    all_child_count.resize(N);
                    all_child_start.resize(N);
                    tree->centers.resize(N);
                    tree->sizes.resize(N);
                }

                std::atomic<bool> failed{false};
                std::mutex error_mutex;
                std::string decode_error;
                auto record_error = [&](std::string msg) {
                    failed.store(true, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (decode_error.empty()) {
                        decode_error = std::move(msg);
                    }
                };

                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, slices.size(), 1),
                    [&](const tbb::blocked_range<size_t>& range) {
                        for (size_t chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
                            if (failed.load(std::memory_order_relaxed)) {
                                return;
                            }
                            const ChunkSlice& slice = slices[chunk_idx];
                            const size_t base = static_cast<size_t>(slice.meta.base);
                            const size_t chunk_count = static_cast<size_t>(slice.meta.count);

                            float* const means = all_means + base * 3;
                            float* const opacity = all_opacity + base;
                            float* const sh0 = all_sh0 + base * 3;
                            float* const scales = all_scales + base * 3;
                            float* const rotation = all_rotation + base * 4;
                            float* const shN =
                                all_shN != nullptr ? all_shN + base * static_cast<size_t>(sh_coeffs) * 3 : nullptr;

                            std::memset(means, 0, chunk_count * 3 * sizeof(float));
                            std::memset(opacity, 0, chunk_count * sizeof(float));
                            std::memset(sh0, 0, chunk_count * 3 * sizeof(float));
                            std::memset(scales, 0, chunk_count * 3 * sizeof(float));
                            std::memset(rotation, 0, chunk_count * 4 * sizeof(float));
                            if (shN != nullptr) {
                                std::memset(shN, 0, chunk_count * static_cast<size_t>(sh_coeffs) * 3 * sizeof(float));
                            }

                            auto err = decode_chunk_properties(
                                data.data(), slice.meta, slice.origin, slice.payload_start,
                                slice.has_payload_prefix, slice.chunk_end, sh_coeffs,
                                means, opacity, sh0, scales, rotation, shN,
                                has_lod_tree ? all_child_count.data() + base : nullptr,
                                has_lod_tree ? all_child_start.data() + base : nullptr);
                            if (err.has_value()) {
                                record_error(std::move(*err));
                                return;
                            }

                            // RAD stores display RGB in the SH0 slot and activated
                            // alpha/scale values; convert back to optimizer domain.
                            for (size_t i = 0; i < chunk_count * 3; ++i) {
                                sh0[i] = (sh0[i] - 0.5f) / SH_C0;
                            }
                            if (!lod_opacity_encoded) {
                                for (size_t i = 0; i < chunk_count; ++i) {
                                    const float a = std::clamp(opacity[i], 1.0e-6f, 1.0f - 1.0e-6f);
                                    opacity[i] = std::log(a / (1.0f - a));
                                }
                            } else {
                                // Spark LOD opacity encoding stores display-space alpha
                                // directly and can exceed 1.0 for dense merged nodes.
                                for (size_t i = 0; i < chunk_count; ++i) {
                                    opacity[i] = std::max(opacity[i], 0.0f);
                                }
                            }
                            if (tree) {
                                for (size_t i = 0; i < chunk_count; ++i) {
                                    tree->centers[base + i] =
                                        glm::vec3(means[i * 3 + 0], means[i * 3 + 1], means[i * 3 + 2]);
                                    const float max_scale =
                                        std::max({scales[i * 3 + 0], scales[i * 3 + 1], scales[i * 3 + 2]});
                                    float expansion = 1.0f;
                                    if (lod_opacity_encoded) {
                                        const float lod_alpha = std::max(opacity[i], 0.0f);
                                        if (lod_alpha > 1.0f) {
                                            const float spark_lod_opacity = std::min(lod_alpha * 4.0f - 3.0f, 5.0f);
                                            expansion = 1.0f + 0.7f * (spark_lod_opacity - 1.0f);
                                        }
                                    }
                                    tree->sizes[base + i] = 2.0f * expansion * max_scale;
                                }
                            }
                            for (size_t i = 0; i < chunk_count * 3; ++i) {
                                scales[i] = std::log(std::max(scales[i], 1.0e-8f));
                            }
                        }
                    });

                if (failed.load()) {
                    return std::unexpected(decode_error);
                }

                SplatData splat_data(
                    max_sh,
                    std::move(means_tensor),
                    std::move(sh0_tensor),
                    std::move(shN_tensor),
                    std::move(scales_tensor),
                    std::move(rotation_tensor),
                    std::move(opacity_tensor),
                    1.0f // scene_scale
                );

                // Attach LOD tree if present
                if (tree && N > 0) {
                    tree->child_count = std::move(all_child_count);
                    tree->child_start = std::move(all_child_start);
                    // RAD files don't store node depths; derive them in one
                    // forward pass (children always follow their parent in the
                    // BFS-ordered layout).
                    tree->lod_level.assign(N, 0);
                    for (size_t i = 0; i < N; ++i) {
                        const std::uint32_t count = tree->child_count[i];
                        if (count == 0) {
                            continue;
                        }
                        const std::uint32_t start = tree->child_start[i];
                        const auto child_level = static_cast<std::uint8_t>(
                            std::min<std::uint32_t>(tree->lod_level[i] + 1u, 255u));
                        for (std::uint32_t c = 0; c < count; ++c) {
                            const size_t child = static_cast<size_t>(start) + c;
                            if (child > i && child < N) {
                                tree->lod_level[child] = child_level;
                            }
                        }
                    }
                    const size_t chunk_count =
                        (N + lfs::core::SplatLodTree::kChunkSplats - 1) /
                        lfs::core::SplatLodTree::kChunkSplats;
                    tree->chunk_to_page.resize(chunk_count);
                    tree->page_to_chunk.resize(chunk_count);
                    std::iota(tree->chunk_to_page.begin(), tree->chunk_to_page.end(), 0u);
                    std::iota(tree->page_to_chunk.begin(), tree->page_to_chunk.end(), 0u);
                    if (source_path != nullptr && !source_path->empty()) {
                        tree->rad_source.path = *source_path;
                        tree->rad_source.chunk_size = meta.chunk_size.value_or(CHUNK_SIZE);
                        tree->rad_source.metadata_bytes = 8 + pad8(meta_size);
                        tree->rad_source.chunks = std::move(chunk_file_ranges);
                    }
                    tree->lod_opacity_encoded = lod_opacity_encoded;
                    splat_data.lod_tree = std::move(tree);
                }

                return std::expected<SplatData, std::string>(std::move(splat_data));
            }
        };

    } // namespace

    // ============================================================================
    // Public API Implementation
    // ============================================================================

    std::expected<SplatData, std::string> load_rad(const std::filesystem::path& filepath) {
        auto start = std::chrono::high_resolution_clock::now();

        LOG_INFO("Loading RAD file: {}", lfs::core::path_to_utf8(filepath));

        // Read file
        std::ifstream in;
        if (!lfs::core::open_file_for_read(filepath, std::ios::binary | std::ios::ate, in)) {
            return std::unexpected(std::format("Failed to open RAD file: {}", lfs::core::path_to_utf8(filepath)));
        }

        const auto size = in.tellg();
        if (size < 0) {
            return std::unexpected(std::format("Failed to read RAD file size: {}", lfs::core::path_to_utf8(filepath)));
        }

        std::vector<uint8_t> data(static_cast<size_t>(size));
        in.seekg(0, std::ios::beg);
        in.read(reinterpret_cast<char*>(data.data()), size);
        in.close();

        if (!in.good()) {
            return std::unexpected(std::format("Failed to read RAD file: {}", lfs::core::path_to_utf8(filepath)));
        }

        // Decode
        RadDecoder decoder;
        auto result = decoder.decode(data, &filepath);

        if (!result) {
            return result;
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);

        LOG_INFO("RAD loaded: {} gaussians with SH degree {} in {}ms",
                 result->size(), result->get_max_sh_degree(), elapsed.count());

        return result;
    }

    std::expected<RadDecodedChunk, std::string> load_rad_chunk(
        const std::filesystem::path& filepath,
        const lfs::core::SplatLodTree::ChunkFileRange& range,
        const int max_sh_degree,
        const bool lod_opacity_encoded) {
        if (range.file_bytes == 0) {
            return std::unexpected("RAD chunk range has zero bytes");
        }
        if (range.file_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            return std::unexpected("RAD chunk range is too large for this platform");
        }

        std::ifstream in;
        if (!lfs::core::open_file_for_read(filepath, std::ios::binary, in)) {
            return std::unexpected(std::format("Failed to open RAD file for chunk read: {}",
                                               lfs::core::path_to_utf8(filepath)));
        }
        in.seekg(static_cast<std::streamoff>(range.file_offset), std::ios::beg);
        if (!in.good()) {
            return std::unexpected(std::format("Failed to seek RAD chunk at offset {} in {}",
                                               range.file_offset,
                                               lfs::core::path_to_utf8(filepath)));
        }

        std::vector<uint8_t> data(static_cast<std::size_t>(range.file_bytes));
        in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!in.good()) {
            return std::unexpected(std::format("Failed to read RAD chunk at offset {} in {}",
                                               range.file_offset,
                                               lfs::core::path_to_utf8(filepath)));
        }

        auto decoded = decode_rad_chunk_buffer(data,
                                               max_sh_degree,
                                               true,
                                               lod_opacity_encoded);
        if (!decoded) {
            return decoded;
        }
        if (range.base != 0 || range.count != 0) {
            if (decoded->base != range.base || decoded->count != range.count) {
                return std::unexpected(std::format(
                    "RAD chunk range mismatch: range base/count={}/{}, decoded base/count={}/{}",
                    range.base,
                    range.count,
                    decoded->base,
                    decoded->count));
            }
        }
        return decoded;
    }

    Result<void> save_rad(const SplatData& splat_data, const RadSaveOptions& options) {
        auto start = std::chrono::high_resolution_clock::now();

        LOG_INFO("Saving RAD file: {}", lfs::core::path_to_utf8(options.output_path));

        int compression_level = options.compression_level;
        if (compression_level != Z_DEFAULT_COMPRESSION &&
            (compression_level < Z_NO_COMPRESSION || compression_level > Z_BEST_COMPRESSION)) {
            LOG_WARN("save_rad: invalid compression_level={} (expected 0..9 or -1), falling back to {}",
                     compression_level, GZ_LEVEL);
            compression_level = GZ_LEVEL;
        }

        // Encode
        RadEncoder encoder(compression_level,
                           options.flip_y,
                           scale_export_progress(options.progress_callback, 0.0f, 0.95f));
        std::vector<uint8_t> data;
        try {
            data = encoder.encode(splat_data);
        } catch (const std::runtime_error& e) {
            if (std::string(e.what()) == "CANCELLED") {
                return make_error(ErrorCode::CANCELLED, "Export cancelled by user");
            }
            throw;
        }

        if (!report_export_progress(options.progress_callback, 0.95f, "Writing RAD")) {
            return make_error(ErrorCode::CANCELLED, "RAD export cancelled", options.output_path);
        }

        if (auto dir_result = ensure_output_parent_directory(options.output_path); !dir_result) {
            return std::unexpected(dir_result.error());
        }

        ScopedAtomicOutputFile atomic_output(options.output_path);
        std::ofstream out;
        if (!lfs::core::open_file_for_write(atomic_output.temp_path(), std::ios::binary | std::ios::out, out)) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              "Failed to open temporary RAD file for writing",
                              atomic_output.temp_path());
        }

        out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        out.close();

        if (!out.good()) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              "Failed to write RAD file", atomic_output.temp_path());
        }

        if (!report_export_progress(options.progress_callback, 1.0f, "RAD export complete")) {
            return make_error(ErrorCode::CANCELLED, "RAD export cancelled", options.output_path);
        }

        if (auto commit_result = atomic_output.commit(); !commit_result) {
            return std::unexpected(commit_result.error());
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);

        // Get file size
        auto file_size = std::filesystem::file_size(options.output_path);
        LOG_INFO("RAD saved: {} gaussians, {:.1f} MB in {}ms",
                 splat_data.size(),
                 static_cast<double>(file_size) / (1024.0 * 1024.0),
                 elapsed.count());

        return {};
    }

    bool rad_paged_load_recommended(const SplatData& data) {
        if (!data.lod_tree || !data.lod_tree->rad_source.valid()) {
            return false;
        }
        const std::size_t logical_chunks = data.lod_tree->chunk_count();
        if (logical_chunks <= 1) {
            return false;
        }
        if (const char* const env = std::getenv("LFS_LOD_PAGE_CAPACITY");
            env != nullptr && env[0] != '\0') {
            try {
                const std::size_t requested = static_cast<std::size_t>(std::stoull(env));
                return std::clamp(requested, std::size_t{1}, logical_chunks) < logical_chunks;
            } catch (...) {
                return false;
            }
        }

        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || free_bytes == 0) {
            return false;
        }
        const auto tensor_bytes = [](const lfs::core::Tensor& t) -> std::size_t {
            return t.is_valid() ? t.bytes() : 0;
        };
        const std::size_t model_bytes =
            tensor_bytes(data.means_raw()) +
            tensor_bytes(data.sh0_raw()) +
            tensor_bytes(data.shN_raw()) +
            tensor_bytes(data.scaling_raw()) +
            tensor_bytes(data.rotation_raw()) +
            tensor_bytes(data.opacity_raw());
        // Stream when full residency would crowd the GPU: the renderer still
        // needs sort scratch, tile buffers, and framebuffers on top.
        const bool paged = model_bytes > free_bytes / 2;
        if (paged) {
            LOG_INFO("RAD paged load recommended: model={:.1f} MB, free VRAM={:.1f} MB",
                     static_cast<double>(model_bytes) / (1024.0 * 1024.0),
                     static_cast<double>(free_bytes) / (1024.0 * 1024.0));
        }
        return paged;
    }

} // namespace lfs::io

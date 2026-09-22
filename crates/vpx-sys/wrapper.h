#include <vpx/vpx_encoder.h>
#include <vpx/vpx_decoder.h>
#include <vpx/vp8cx.h>
#include <vpx/vp8dx.h>

/* Helpers for what bindgen cannot express: the ABI-versioned init macros and the variadic
 * vpx_codec_control_(). Implemented in shim.c. */
vpx_codec_err_t fflv_vp9_enc_init(vpx_codec_ctx_t *ctx, const vpx_codec_enc_cfg_t *cfg, vpx_codec_flags_t flags);
vpx_codec_err_t fflv_vp9_dec_init(vpx_codec_ctx_t *ctx, const vpx_codec_dec_cfg_t *cfg);
vpx_codec_err_t fflv_vpx_control_int(vpx_codec_ctx_t *ctx, int ctrl_id, int value);

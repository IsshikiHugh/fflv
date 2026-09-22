#include "wrapper.h"

vpx_codec_err_t fflv_vp9_enc_init(vpx_codec_ctx_t *ctx, const vpx_codec_enc_cfg_t *cfg, vpx_codec_flags_t flags) {
    return vpx_codec_enc_init(ctx, vpx_codec_vp9_cx(), cfg, flags);
}

vpx_codec_err_t fflv_vp9_dec_init(vpx_codec_ctx_t *ctx, const vpx_codec_dec_cfg_t *cfg) {
    return vpx_codec_dec_init(ctx, vpx_codec_vp9_dx(), cfg, 0);
}

vpx_codec_err_t fflv_vpx_control_int(vpx_codec_ctx_t *ctx, int ctrl_id, int value) {
    return vpx_codec_control_(ctx, ctrl_id, value);
}

const std = @import("std");
const Camera = @This();

pub const Capability = struct {
    driver: [16]u8,
    card: [32]u8,
    bus_info: [32]u8,
    version: u32,
    device_caps: u32,
    reserved: [3]u32,
};

pub const Frame = struct {
    buffer: struct {
        ptr: [*]u8,
        len: u8,
    },
    tag: u32,
    fmt: struct {
        pix: struct {
            width: u32,
            height: u32,
            pixelformat: u32,
            field: u32,
            bytesperline: u32,
            sizeimage: u32,
            colorspace: u32,
            priv: u32,
            flag: u32,
            enc: union {
                ycbcr: u32,
                hsv: u32,
            },
            quantization: u32,
            xfer_func: u32,
        },
        pix_mplane: struct {
            width: u32,
            height: u32,
            pixelformat: u32,
            field: u32,
            plane_fmt: [8]struct {
                sizeimage: u32,
                bytesperline: u32,
                reserved: [6]u16,
            },
        },
        window: struct {
            rect: struct {
                left: u32,
                top: u32,
                width: u32,
                height: u32,
            },
            field: u32,
            chromakey: u32,
            clips: *struct {
                c: struct {
                    left: u32,
                    top: u32,
                    width: u32,
                    height: u32,
                },
            },
        },
    },
};

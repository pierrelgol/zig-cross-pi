const std = @import("std");

const c = @cImport({
    @cInclude("linux/videodev2.h");
    @cInclude("sys/ioctl.h");
    @cInclude("fcntl.h");
    @cInclude("unistd.h");
    @cInclude("sys/mman.h");
});

const Buffer = struct {
    start: [*]u8,
    length: usize,
};

pub const Point = struct { x: i32, y: i32 };

pub const Moments = struct {
    m00: f64,
    m10: f64,
    m01: f64,
    m20: f64,
    m02: f64,
    m11: f64,
};

inline fn absI32(v: i32) i32 {
    return if (v >= 0) v else -v;
}

pub fn computeMoments(points: []const Point) Moments {
    var mu00: f64 = 0;
    var mu10: f64 = 0;
    var mu01: f64 = 0;
    var mu20: f64 = 0;
    var mu02: f64 = 0;
    var mu11: f64 = 0;

    for (points) |p| {
        mu00 += 1;
        const fx: f64 = @floatFromInt(p.x);
        const fy: f64 = @floatFromInt(p.y);
        mu10 += fx;
        mu01 += fy;
        mu20 += fx * fx;
        mu02 += fy * fy;
        mu11 += fx * fy;
    }

    return .{ .m00 = mu00, .m10 = mu10, .m01 = mu01, .m20 = mu20, .m02 = mu02, .m11 = mu11 };
}

/// mask: binary (0 or 255), row-major
/// out_points: flat storage for all contour pixels
/// out_starts: starts[ci] = start index in out_points; includes sentinel at starts[contour_count] == total_points
/// work_visited: length must be w*h, values set to 0 initially; function sets to 1 for visited
/// work_stack: temp stack for flood-fill; length >= w*h recommended
pub fn findContours(
    mask: []const u8,
    w: usize,
    h: usize,
    out_points: []Point,
    out_starts: []usize,
    work_visited: []u8,
    work_stack: []Point,
) !usize {
    if (work_visited.len != w * h) return error.BadVisitedBuffer;
    if (out_starts.len == 0) return error.NoSpaceForStarts;

    // clear visited
    @memset(work_visited, 0);

    var point_count: usize = 0;
    var contour_count: usize = 0;

    var sp: usize = 0; // stack pointer

    var y: usize = 0;
    while (y < h) : (y += 1) {
        var x: usize = 0;
        while (x < w) : (x += 1) {
            const idx = y * w + x;
            if (mask[idx] != 0 and work_visited[idx] == 0) {
                if (contour_count + 1 >= out_starts.len) // +1 for sentinel later
                    return error.TooManyContours;

                out_starts[contour_count] = point_count;
                contour_count += 1;

                if (sp >= work_stack.len) return error.StackOverflow;
                work_stack[sp] = .{ .x = @as(i32, @intCast(x)), .y = @as(i32, @intCast(y)) };
                sp += 1;
                work_visited[idx] = 1;

                // iterative DFS (8-connected)
                while (sp > 0) {
                    sp -= 1;
                    const p = work_stack[sp];

                    if (point_count >= out_points.len) return error.OutBufferFull;
                    var is_boundary = false;
                    var dyy: isize = -1;
                    while (dyy <= 1) : (dyy += 1) {
                        var dx: isize = -1;
                        while (dx <= 1) : (dx += 1) {
                            if (dx == 0 and dyy == 0) continue;
                            const nx: isize = @as(isize, p.x) + dx;
                            const ny: isize = @as(isize, p.y) + dyy;
                            if (nx < 0 or ny < 0 or nx >= @as(isize, @intCast(w)) or ny >= @as(isize, @intCast(h))) continue;

                            const nidx = @as(usize, @intCast(ny)) * w + @as(usize, @intCast(nx));
                            if (mask[nidx] == 0) {
                                is_boundary = true;
                            }
                        }
                    }

                    if (is_boundary) {
                        out_points[point_count] = p;
                        point_count += 1;
                    }

                    var dy: isize = -1;
                    while (dy <= 1) : (dy += 1) {
                        var dx: isize = -1;
                        while (dx <= 1) : (dx += 1) {
                            if (dx == 0 and dy == 0) continue;

                            const nx: isize = @as(isize, p.x) + dx;
                            const ny: isize = @as(isize, p.y) + dy;
                            if (nx < 0 or ny < 0) continue;
                            if (nx >= @as(isize, @intCast(w)) or ny >= @as(isize, @intCast(h))) continue;

                            const ux = @as(usize, @intCast(nx));
                            const uy = @as(usize, @intCast(ny));
                            const nidx = uy * w + ux;

                            if (mask[nidx] != 0 and work_visited[nidx] == 0) {
                                if (sp >= work_stack.len) return error.StackOverflow;
                                work_stack[sp] = .{ .x = @as(i32, @intCast(ux)), .y = @as(i32, @intCast(uy)) };
                                sp += 1;
                                work_visited[nidx] = 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // sentinel marking end of last contour
    out_starts[contour_count] = point_count;
    return contour_count;
}

pub fn putPixel(buf: []u8, w: usize, h: usize, x: i32, y: i32, r: u8, g: u8, b: u8) void {
    if (x < 0 or y < 0) return;
    if (x >= @as(i32, @intCast(w)) or y >= @as(i32, @intCast(h))) return;
    const ux = @as(usize, @intCast(x));
    const uy = @as(usize, @intCast(y));
    const idx = (uy * w + ux) * 3;
    if (idx + 2 >= buf.len) return; // safety
    buf[idx + 0] = r;
    buf[idx + 1] = g;
    buf[idx + 2] = b;
}

pub fn drawLine(buf: []u8, w: usize, h: usize, x0: i32, y0: i32, x1: i32, y1: i32, r: u8, g: u8, b: u8) void {
    var cx = x0;
    var cy = y0;
    const dx = absI32(x1 - x0);
    const sx: i32 = if (x0 < x1) 1 else -1;
    const dy = -absI32(y1 - y0);
    const sy: i32 = if (y0 < y1) 1 else -1;
    var err: i32 = dx + dy;

    while (true) {
        putPixel(buf, w, h, cx, cy, r, g, b);
        if (cx == x1 and cy == y1) break;

        const e2 = err * 2;
        if (e2 >= dy) {
            err += dy;
            cx += sx;
        }
        if (e2 <= dx) {
            err += dx;
            cy += sy;
        }
    }
}

/// Draws a polyline for each contour, closing the loop.
pub fn drawContours(
    buf: []u8,
    w: usize,
    h: usize,
    points: []const Point,
    starts: []const usize,
    contour_count: usize,
    r: u8,
    g: u8,
    b: u8,
) void {
    var ci: usize = 0;
    while (ci < contour_count) : (ci += 1) {
        const start = starts[ci];
        const end = starts[ci + 1];
        if (end <= start + 1) continue;

        const contour = points[start..end];
        var i: usize = 0;
        while (i < contour.len) : (i += 1) {
            const p0 = contour[i];
            const p1 = contour[(i + 1) % contour.len]; // wrap to close
            drawLine(buf, w, h, p0.x, p0.y, p1.x, p1.y, r, g, b);
        }
    }
}

fn xioctl(fd: c_int, req: c_ulong, arg: *anyopaque) !void {
    if (c.ioctl(fd, req, arg) < 0) return error.IoctlFailed;
}

fn yuyvToRgb24(dst: []u8, src: []const u8, _: usize, _: usize) void {
    var di: usize = 0;
    var si: usize = 0;

    while (si + 3 < src.len and di + 5 < dst.len) {
        const y0: i32 = src[si];
        const u: i32 = src[si + 1] -| 128;
        const y1: i32 = src[si + 2];
        const v: i32 = src[si + 3] -| 128;
        si += 4;

        const r0 = std.math.clamp(y0 + @divFloor(1402 * v, 1000), 0, 255);
        const g0 = std.math.clamp(y0 - @divFloor(344 * u, 1000) - @divFloor(714 * v, 1000), 0, 255);
        const b0 = std.math.clamp(y0 + @divFloor(1772 * u, 1000), 0, 255);

        const r1 = std.math.clamp(y1 + @divFloor(1402 * v, 1000), 0, 255);
        const g1 = std.math.clamp(y1 - @divFloor(344 * u, 1000) - @divFloor(714 * v, 1000), 0, 255);
        const b1 = std.math.clamp(y1 + @divFloor(1772 * u, 1000), 0, 255);

        dst[di] = @as(u8, @intCast(r0));
        dst[di + 1] = @as(u8, @intCast(g0));
        dst[di + 2] = @as(u8, @intCast(b0));

        dst[di + 3] = @as(u8, @intCast(r1));
        dst[di + 4] = @as(u8, @intCast(g1));
        dst[di + 5] = @as(u8, @intCast(b1));

        di += 6;
    }
}

fn threshold(in: []const u8, out: []u8, thresh: u8) void {
    for (in, 0..) |px, i| {
        out[i] = if (px > thresh) 255 else 0;
    }
}

fn dilate(src: []const u8, dst: []u8, w: usize, h: usize) void {
    for (0..h) |y| {
        for (0..w) |x| {
            var max: u8 = 0;
            var dy: isize = -1;
            while (dy <= 1) : (dy += 1) {
                var dx: isize = -1;
                while (dx <= 1) : (dx += 1) {
                    const nx: isize = @as(isize, @intCast(x)) + dx;
                    const ny: isize = @as(isize, @intCast(y)) + dy;
                    if (nx >= 0 and ny >= 0 and nx < @as(isize, @intCast(w)) and ny < @as(isize, @intCast(h))) {
                        const val = src[@as(usize, @intCast(ny)) * w + @as(usize, @intCast(nx))];
                        if (val > max) max = val;
                    }
                }
            }
            dst[y * w + x] = max;
        }
    }
}

fn erode(src: []const u8, dst: []u8, w: usize, h: usize) void {
    for (0..h) |y| {
        for (0..w) |x| {
            var min: u8 = 255;
            var dy: isize = -1;
            while (dy <= 1) : (dy += 1) {
                var dx: isize = -1;
                while (dx <= 1) : (dx += 1) {
                    const nx: isize = @as(isize, @intCast(x)) + dx;
                    const ny: isize = @as(isize, @intCast(y)) + dy;
                    if (nx >= 0 and ny >= 0 and nx < @as(isize, @intCast(w)) and ny < @as(isize, @intCast(h))) {
                        const val = src[@as(usize, @intCast(ny)) * w + @as(usize, @intCast(nx))];
                        if (val < min) min = val;
                    }
                }
            }
            dst[y * w + x] = min;
        }
    }
}

fn computeCircularity(mu: Moments) f64 {
    if (mu.m00 == 0) return 0;

    const sx = mu.m10 / mu.m00;
    const sy = mu.m01 / mu.m00;

    const cm0 = mu.m20 / mu.m00 - sx * sx;
    const cm1 = mu.m11 / mu.m00 - sx * sy;
    const cm2 = mu.m02 / mu.m00 - sy * sy;

    const det = (cm0 + cm2) * (cm0 + cm2) - 4 * (cm0 * cm2 - cm1 * cm1);
    const f0 = ((cm0 + cm2) + @sqrt(@max(0.0, det))) / 2.0;
    const f1 = ((cm0 + cm2) - @sqrt(@max(0.0, det))) / 2.0;

    return std.math.pi * 4.0 * @sqrt(f0) * @sqrt(f1) / mu.m00;
}

fn computeCenter(points: []const Point) Point {
    if (points.len == 0) return .{ .x = 0, .y = 0 };

    var sumx: i64 = 0;
    var sumy: i64 = 0;
    for (points) |p| {
        sumx += p.x;
        sumy += p.y;
    }
    const cx = @divTrunc(sumx, @as(i64, @intCast(points.len)));
    const cy = @divTrunc(sumy, @as(i64, @intCast(points.len)));
    return .{ .x = @as(i32, @intCast(cx)), .y = @as(i32, @intCast(cy)) };
}

const Pose = struct {
    posX: f32,
    posY: f32,
    roll: f32,
    ok: bool,
};

fn fitLineAndPose(
    centers: []const Point,
    w: usize,
    h: usize,
    capture_zone_enlargement: f32,
) Pose {
    if (centers.len < 2) return .{ .posX = 0, .posY = 0, .roll = 0, .ok = false };

    // mean
    var meanX: f64 = 0;
    var meanY: f64 = 0;
    for (centers) |p| {
        meanX += @floatFromInt(p.x);
        meanY += @floatFromInt(p.y);
    }
    meanX /= @floatFromInt(centers.len);
    meanY /= @floatFromInt(centers.len);

    // covariance
    var Sxx: f64 = 0;
    var Sxy: f64 = 0;
    for (centers) |p| {
        const dx: f64 = @as(f64, @floatFromInt(p.x)) - meanX;
        const dy: f64 = @as(f64, @floatFromInt(p.y)) - meanY;
        Sxx += dx * dx;
        Sxy += dx * dy;
    }

    var dirx: f64 = 1.0;
    var diry: f64 = 0.0;
    if (@abs(Sxx) > 1e-6) {
        const slope = Sxy / Sxx;
        dirx = 1.0;
        diry = slope;
        const len = @sqrt(dirx * dirx + diry * diry);
        if (len > 1e-6) {
            dirx /= len;
            diry /= len;
        }
    } else {
        dirx = 0.0;
        diry = 1.0;
    }

    // project points
    var minProj: f64 = std.math.floatMax(f64);
    var maxProj: f64 = -std.math.floatMax(f64);
    for (centers) |p| {
        const dx = @as(f64, @floatFromInt(p.x)) - meanX;
        const dy = @as(f64, @floatFromInt(p.y)) - meanY;
        const proj = dx * dirx + dy * diry;
        if (proj < minProj) minProj = proj;
        if (proj > maxProj) maxProj = proj;
    }

    const startx = meanX + minProj * dirx;
    const starty = meanY + minProj * diry;
    const endx = meanX + maxProj * dirx;
    const endy = meanY + maxProj * diry;

    const rollAngle = std.math.atan2(endy - starty, endx - startx);

    const segCx = 0.5 * (startx + endx);
    const segCy = 0.5 * (starty + endy);
    const diffx = segCx - @as(f64, @floatFromInt(w)) * 0.5;
    const diffy = segCy - @as(f64, @floatFromInt(h)) * 0.5;

    const cosA = @cos(-rollAngle);
    const sinA = @sin(-rollAngle);
    var rx = diffx * cosA - diffy * sinA;
    var ry = diffx * sinA + diffy * cosA;

    const scale = capture_zone_enlargement * @as(f64, @floatCast(@min(@as(f64, @floatFromInt(w)), @as(f64, @floatFromInt(h))))) / @sqrt(2.0);
    rx /= scale;
    ry /= scale;

    const posX = std.math.clamp(@as(f32, @floatCast(rx + 0.5)), 0.0, 1.0);
    const posY = std.math.clamp(@as(f32, @floatCast(ry + 0.5)), 0.0, 1.0);

    return .{ .posX = posX, .posY = posY, .roll = @as(f32, @floatCast(rollAngle)), .ok = true };
}

// Add this helper: convert RGB24 to grayscale
fn rgbToGray(gray: []u8, rgb: []const u8, _: usize, _: usize) void {
    var i: usize = 0;
    var j: usize = 0;
    while (i < rgb.len and j < gray.len) {
        const r: i32 = rgb[i];
        const g: i32 = rgb[i + 1];
        const b: i32 = rgb[i + 2];
        const y: i32 = @divFloor((299 * r + 587 * g + 114 * b), 1000);
        gray[j] = @as(u8, @intCast(std.math.clamp(y, 0, 255)));
        i += 3;
        j += 1;
    }
}

pub fn drawCircle(buf: []u8, w: usize, h: usize, center: Point, radius: i32, r: u8, g: u8, b: u8) void {
    var x: i32 = -radius;
    while (x <= radius) : (x += 1) {
        var y: i32 = -radius;
        while (y <= radius) : (y += 1) {
            if (x * x + y * y <= radius * radius) {
                putPixel(buf, w, h, center.x + x, center.y + y, r, g, b);
            }
        }
    }
}

pub fn main() !void {
    var gpa_instance: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa_instance.deinit();
    const allocator = gpa_instance.allocator();

    // Open device
    const fd = c.open("/dev/video0", c.O_RDWR, @as(u8, 0));
    if (fd < 0) return error.OpenFailed;

    // Set format
    var fmt: c.struct_v4l2_format = std.mem.zeroes(c.struct_v4l2_format);
    fmt.type = c.V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = 640;
    fmt.fmt.pix.height = 480;
    fmt.fmt.pix.pixelformat = c.V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = c.V4L2_FIELD_NONE;
    try xioctl(fd, c.VIDIOC_S_FMT, &fmt);

    // Request buffers
    var req: c.struct_v4l2_requestbuffers = std.mem.zeroes(c.struct_v4l2_requestbuffers);
    req.count = 1;
    req.type = c.V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = c.V4L2_MEMORY_MMAP;
    try xioctl(fd, c.VIDIOC_REQBUFS, &req);

    // Query buffer
    var bufinfo: c.struct_v4l2_buffer = std.mem.zeroes(c.struct_v4l2_buffer);
    bufinfo.type = c.V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bufinfo.memory = c.V4L2_MEMORY_MMAP;
    bufinfo.index = 0;
    try xioctl(fd, c.VIDIOC_QUERYBUF, &bufinfo);

    // mmap buffer
    const ptr = c.mmap(
        null,
        bufinfo.length,
        c.PROT_READ | c.PROT_WRITE,
        c.MAP_SHARED,
        fd,
        @as(c.__off_t, @intCast(bufinfo.m.offset)),
    );
    if (ptr == c.MAP_FAILED) return error.MmapFailed;

    const buffer = Buffer{
        .start = @as([*]u8, @ptrCast(ptr)),
        .length = bufinfo.length,
    };

    // Queue buffer
    try xioctl(fd, c.VIDIOC_QBUF, &bufinfo);

    // Start streaming
    var t: c_int = c.V4L2_BUF_TYPE_VIDEO_CAPTURE;
    try xioctl(fd, c.VIDIOC_STREAMON, &t);

    const w: usize = 640;
    const h: usize = 480;
    const rgb_size = w * h * 3;
    const rgb_buf = try allocator.alloc(u8, rgb_size);
    defer allocator.free(rgb_buf);

    // workspace buffers
    var gray_buf: [w * h]u8 = undefined;
    var thres_buf: [w * h]u8 = undefined;
    var morph1: [w * h]u8 = undefined;
    var morph2: [w * h]u8 = undefined;
    var points_buf: [200_000]Point = undefined;
    var starts_buf: [2001]usize = undefined; // +1 sentinel
    var visited_buf: [w * h]u8 = undefined;
    var stack_buf: [200_000]Point = undefined;
    var centers_buf: [512]Point = undefined;

    // Capture multiple frames
    var frame_idx: usize = 0;
    while (frame_idx < 1000) : (frame_idx += 1) {
        // Dequeue
        try xioctl(fd, c.VIDIOC_DQBUF, &bufinfo);

        // Convert YUYV → RGB
        yuyvToRgb24(rgb_buf, buffer.start[0..buffer.length], w, h);

        // === pipeline ===
        rgbToGray(gray_buf[0..], rgb_buf, w, h);
        threshold(gray_buf[0..], thres_buf[0..], 200);
        dilate(thres_buf[0..], morph1[0..], w, h);
        erode(morph1[0..], morph2[0..], w, h);

        const contour_count = try findContours(
            morph2[0..],
            w,
            h,
            points_buf[0..],
            starts_buf[0..],
            visited_buf[0..],
            stack_buf[0..],
        );

        var center_count: usize = 0;
        var ci: usize = 0;
        while (ci < contour_count) : (ci += 1) {
            const a = starts_buf[ci];
            const b = starts_buf[ci + 1];
            const contour = points_buf[a..b];

            // Compute bounding box
            var minx: i32 = std.math.maxInt(i32);
            var miny: i32 = std.math.maxInt(i32);
            var maxx: i32 = std.math.minInt(i32);
            var maxy: i32 = std.math.minInt(i32);
            for (contour) |p| {
                if (p.x < minx) minx = p.x;
                if (p.y < miny) miny = p.y;
                if (p.x > maxx) maxx = p.x;
                if (p.y > maxy) maxy = p.y;
            }
            const bw = maxx - minx + 1;
            const bh = maxy - miny + 1;

            // Reject if outside pixel size range
            if (bw > 80 or bh > 80 or bw < 5 or bh < 5) continue;

            // Reject if too elongated (aspect ratio check)
            const aspect = @as(f64, @floatFromInt(bw)) / @as(f64, @floatFromInt(bh));
            if (aspect < 0.7 or aspect > 1.3) continue;

            // --- circularity ---
            const mu = computeMoments(contour);
            const circ = computeCircularity(mu);
            if (@abs(circ - 1.0) > 0.05) continue;

            // --- center ---
            if (center_count < centers_buf.len) {
                centers_buf[center_count] = computeCenter(contour);
                center_count += 1;
            }
        }

        // Debug: how many circles detected this frame
        std.debug.print("Frame {d}: {d} circle(s) found\n", .{ frame_idx, center_count });

        const pose = fitLineAndPose(centers_buf[0..center_count], w, h, 1.5);
        if (pose.ok) {
            std.debug.print("Frame {d}: posX={d:.3} posY={d:.3} roll={d:.3}\n", .{ frame_idx, pose.posX, pose.posY, pose.roll });
        } else {
            std.debug.print("Frame {d}: no valid pose\n", .{frame_idx});
        }

        // === draw overlays ===
        drawContours(rgb_buf, w, h, points_buf[0..], starts_buf[0..], contour_count, 255, 0, 0);
        for (centers_buf[0..center_count]) |cp| {
            drawCircle(rgb_buf, w, h, cp, 3, 0, 255, 0);
        }

        // Save frame with overlays
        var filename_buf: [64]u8 = undefined;
        const filename = try std.fmt.bufPrint(&filename_buf, "/tmp/frame.ppm", .{});
        var file = try std.fs.cwd().createFile(filename, .{ .truncate = true, .lock = .exclusive });
        defer file.close();

        var buff: [4096]u8 = undefined;
        var bw = file.writer(&buff);
        var fw = &bw.interface;

        try fw.print("P6\n{} {}\n255\n", .{ w, h });
        try fw.writeAll(rgb_buf);
        try fw.flush();

        // Requeue
        try xioctl(fd, c.VIDIOC_QBUF, &bufinfo);
    }

    // Stop streaming
    try xioctl(fd, c.VIDIOC_STREAMOFF, &t);

    // Cleanup
    _ = c.munmap(ptr, bufinfo.length);
    _ = c.close(fd);
}

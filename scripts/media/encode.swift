// Turns the frames recorded by capture.sh into docs/media/demo.mp4 and docs/media/demo.gif using only what ships
// with macOS (AVFoundation and ImageIO), so no ffmpeg is needed.
//
//   swiftc -O scripts/media/encode.swift -o /tmp/encode && /tmp/encode FRAMES_DIR OUT_DIR
//
// Each frame gets a small caption in the window's empty title bar: it says the clip is sped up and how much real
// time had passed, taken from the capture time of the frame.
import AppKit
import AVFoundation
import CoreVideo
import ImageIO
import UniformTypeIdentifiers

let argv = CommandLine.arguments
guard argv.count >= 3 else {
    FileHandle.standardError.write(Data("usage: encode FRAMES_DIR OUT_DIR [mp4_fps=12] [gif_frames=120] [tail=0]\n".utf8))
    exit(2)
}
let framesDir = URL(fileURLWithPath: argv[1])
let outDir = URL(fileURLWithPath: argv[2])
let mp4Fps = argv.count > 3 ? Int(argv[3]) ?? 12 : 12
let gifFrameBudget = argv.count > 4 ? Int(argv[4]) ?? 120 : 120
/// The last `tail` frames are the interactive part (chat, other screens); they linger instead of racing by.
let tail = argv.count > 5 ? Int(argv[5]) ?? 0 : 0
let tailRepeat = 3
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

// MARK: frames

let names = try FileManager.default.contentsOfDirectory(atPath: framesDir.path).filter { $0.hasSuffix(".jpg") }.sorted()
guard !names.isEmpty else { print("no frames in \(framesDir.path)"); exit(1) }
let stamps: [Date] = names.map {
    ((try? FileManager.default.attributesOfItem(atPath: framesDir.appendingPathComponent($0).path))?[.modificationDate] as? Date) ?? Date()
}
// Elapsed real time per frame. A pause longer than a few seconds (between two recording sessions) counts as one.
var elapsed: [TimeInterval] = [0]
for i in 1..<max(1, names.count) { elapsed.append(elapsed[i - 1] + min(3.0, max(0, stamps[i].timeIntervalSince(stamps[i - 1])))) }
let realSeconds = elapsed.last ?? 0

func load(_ name: String) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(framesDir.appendingPathComponent(name) as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(src, 0, nil)
}

/// Scales a frame to `width` pixels wide and writes the caption into its title bar.
func render(_ image: CGImage, width: Int, caption: String) -> CGImage? {
    let height = Int((Double(image.height) * Double(width) / Double(image.width)).rounded()) & ~1
    let space = CGColorSpace(name: CGColorSpace.sRGB)!
    guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0, space: space,
                              bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
    ctx.interpolationQuality = .high
    ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

    let fontSize = max(9.0, Double(width) * 0.0105)
    let attrs: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: fontSize, weight: .medium),
        .foregroundColor: NSColor(calibratedWhite: 0.28, alpha: 1),
    ]
    let text = NSAttributedString(string: caption, attributes: attrs)
    let size = text.size()
    let barHeight = Double(height) * 0.037
    let nsctx = NSGraphicsContext(cgContext: ctx, flipped: false)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = nsctx
    text.draw(at: NSPoint(x: Double(width) - size.width - Double(width) * 0.014, y: Double(height) - barHeight / 2 - size.height / 2))
    NSGraphicsContext.restoreGraphicsState()
    return ctx.makeImage()
}

func clock(_ seconds: TimeInterval) -> String {
    let s = Int(max(0, seconds))
    return String(format: "%d:%02d", s / 60, s % 60)
}

func caption(_ i: Int) -> String {
    "Sped up. Real time \(clock(elapsed[i])) of \(clock(realSeconds))"
}

// MARK: MP4

func writeMP4() throws {
    let url = outDir.appendingPathComponent("demo.mp4")
    try? FileManager.default.removeItem(at: url)
    let width = 1280
    guard let first = load(names[0]), let probe = render(first, width: width, caption: "") else { return }
    let height = probe.height
    let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: width, AVVideoHeightKey: height,
        AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 3_500_000, AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel],
    ])
    input.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
    ])
    writer.add(input)
    writer.startWriting()
    writer.startSession(atSourceTime: .zero)

    var frame = 0
    let hold = mp4Fps * 2 // dwell on the last frame
    // play list: every frame once, the tail frames `tailRepeat` times, then a dwell on the last frame
    var order: [Int] = Array(repeating: 0, count: mp4Fps * 3 / 2) // let the first screen be read before the speed-up starts
    for i in 0..<names.count { order += Array(repeating: i, count: i >= names.count - tail ? tailRepeat : 1) }
    order += Array(repeating: names.count - 1, count: hold)
    for i in order {
        guard let img = load(names[i]), let out = render(img, width: width, caption: caption(i)) else { continue }
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, nil, &buffer)
        guard let pb = buffer else { continue }
        CVPixelBufferLockBaseAddress(pb, [])
        if let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: width, height: height, bitsPerComponent: 8,
                               bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpace(name: CGColorSpace.sRGB)!,
                               bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue) {
            ctx.draw(out, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        CVPixelBufferUnlockBaseAddress(pb, [])
        while !input.isReadyForMoreMediaData { usleep(2000) }
        adaptor.append(pb, withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: CMTimeScale(mp4Fps)))
        frame += 1
    }
    input.markAsFinished()
    let done = DispatchSemaphore(value: 0)
    writer.finishWriting { done.signal() }
    done.wait()
    if writer.status != .completed { print("mp4 failed: \(String(describing: writer.error))"); exit(1) }
    let bytes = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
    print("demo.mp4  \(frame) frames at \(mp4Fps) fps, \(width)x\(height), \(bytes / 1024) KiB")
}

// MARK: GIF

func writeGIF() throws {
    let url = outDir.appendingPathComponent("demo.gif")
    try? FileManager.default.removeItem(at: url)
    let width = 960
    // pick evenly spaced frames so the clip stays a reasonable size
    let head = names.count - tail
    let step = max(1, Int((Double(head) / Double(gifFrameBudget)).rounded(.up)))
    var picks = Array(stride(from: 0, to: head, by: step))
    picks += Array(stride(from: head, to: names.count, by: 2))
    if picks.last != names.count - 1 { picks.append(names.count - 1) }
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.gif.identifier as CFString, picks.count, nil) else { return }
    CGImageDestinationSetProperties(dest, [kCGImagePropertyGIFDictionary: [kCGImagePropertyGIFLoopCount: 0]] as CFDictionary)
    let delay = 0.14
    for (k, i) in picks.enumerated() {
        guard let img = load(names[i]), let out = render(img, width: width, caption: caption(i)) else { continue }
        let isLast = k == picks.count - 1
        let frameDelay = isLast ? 2.5 : (k == 0 ? 1.5 : (i >= head ? delay * 3 : delay))
        CGImageDestinationAddImage(dest, out, [kCGImagePropertyGIFDictionary: [kCGImagePropertyGIFDelayTime: frameDelay]] as CFDictionary)
    }
    guard CGImageDestinationFinalize(dest) else { print("gif failed"); exit(1) }
    let bytes = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
    print("demo.gif  \(picks.count) frames, \(width) px wide, \(bytes / 1024) KiB")
}

print("\(names.count) frames spanning \(clock(realSeconds)) of real time")
try writeMP4()
try writeGIF()

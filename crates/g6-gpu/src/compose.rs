//! Two presented planes as one image: the console's display puts the second
//! (the interface) over the first (the scene), each scaled to the screen.
//! The title presents so on Series S.
//!
//! `G6_PLANES=0` ([`on`]) shows the first plane alone. A present of one plane
//! is not affected either way.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

use crate::host::Host;

const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=2), visibility=SHADER_VISIBILITY_PIXEL), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP)"
Texture2D<float4> scene : register(t0);
Texture2D<float4> over : register(t1);
SamplerState smooth_ : register(s0);
struct V { float4 at : SV_Position; float2 uv : TEXCOORD0; };
[RootSignature(RS)]
V vs(uint id : SV_VertexID) {
    V v;
    v.uv = float2((id << 1) & 2, id & 2);
    v.at = float4(v.uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return v;
}
[RootSignature(RS)]
float4 ps(V v) : SV_Target {
    float4 s = scene.SampleLevel(smooth_, v.uv, 0);
    float4 o = over.SampleLevel(smooth_, v.uv, 0);
    return float4(saturate(o.rgb + s.rgb * (1 - saturate(o.a))), 1);
}
"#;

/// Whether a second presented plane is composed over the first
/// (`G6_PLANES`; on by default, `0` turns it off).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("G6_PLANES").is_none_or(|v| v != "0"))
}

pub(crate) struct Composer {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    views: Vec<ID3D12DescriptorHeap>,
    step: u32,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Composer {}
// SAFETY: as above.
unsafe impl Sync for Composer {}

impl Composer {
    /// A composer into render targets of `format`.
    pub(crate) fn new(host: &Host, format: DXGI_FORMAT) -> windows::core::Result<Self> {
        let vs = crate::host::compile(SOURCE, s!("vs"), s!("vs_5_1"))?;
        let ps = crate::host::compile(SOURCE, s!("ps"), s!("ps_5_1"))?;
        let device = &host.device;
        let code = |b: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: b.as_ptr().cast(),
            BytecodeLength: b.len(),
        };
        // SAFETY: the compiled bytes and the root signature live through the
        // calls (the description borrows the signature without releasing it).
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &vs)?;
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs),
                PS: code(&ps),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            desc.RTVFormats[0] = format;
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            let pso = device.CreateGraphicsPipelineState(&desc)?;
            Ok(Composer {
                root,
                pso,
                views: Vec::new(),
                step: device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
            })
        }
    }

    /// Records into `list` the draw of the two planes into `back`. Each
    /// presenter list slot owns its own pair of views: several presents may
    /// be in the air, and a slot is used again only once its list has run.
    ///
    /// The planes are in the pixel shader resource state, `back` is a render
    /// target.
    pub(crate) fn record(
        &mut self,
        host: &Host,
        slot: usize,
        list: &ID3D12GraphicsCommandList,
        planes: [(&ID3D12Resource, DXGI_FORMAT); 2],
        back: &ID3D12Resource,
        (width, height): (u32, u32),
    ) -> windows::core::Result<()> {
        while self.views.len() <= slot {
            let desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            // SAFETY: plain object creation.
            let heap = unsafe { host.device.CreateDescriptorHeap(&desc)? };
            self.views.push(heap);
        }
        let views = &self.views[slot];
        // SAFETY: an open list; the views are made in slots no frame in flight
        // reads; live textures.
        unsafe {
            let (cpu, gpu) = (
                views.GetCPUDescriptorHandleForHeapStart(),
                views.GetGPUDescriptorHandleForHeapStart(),
            );
            for (i, plane) in planes.iter().enumerate() {
                let at = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu.ptr + i * self.step as usize,
                };
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: plane.1,
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_SRV {
                            MipLevels: 1,
                            ..Default::default()
                        },
                    },
                };
                host.device.CreateShaderResourceView(plane.0, Some(&desc), at);
            }
            let rtv = host.rtv(Some(back), None);
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.SetDescriptorHeaps(&[Some(views.clone())]);
            list.SetGraphicsRootSignature(&self.root);
            list.SetPipelineState(&self.pso);
            list.SetGraphicsRootDescriptorTable(0, gpu);
            list.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                Width: width as f32,
                Height: height as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            }]);
            list.DrawInstanced(3, 1, 0, 0);
        }
        Ok(())
    }
}

/// The format a display plane is sampled through, keeping a typed resource's
/// own; typeless resources get the matching display type. `None` for what
/// cannot be sampled as a plane.
pub(crate) fn view(d: &D3D12_RESOURCE_DESC) -> Option<DXGI_FORMAT> {
    if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || d.DepthOrArraySize != 1
        || d.SampleDesc.Count != 1
        || d.Flags.contains(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)
    {
        return None;
    }
    match d.Format.0 {
        9 => Some(DXGI_FORMAT_R16G16B16A16_FLOAT),
        23 => Some(DXGI_FORMAT_R10G10B10A2_UNORM),
        27 => Some(DXGI_FORMAT_R8G8B8A8_UNORM),
        90 => Some(DXGI_FORMAT_B8G8R8A8_UNORM),
        10 | 11 | 13 | 24 | 26 | 28 | 29 | 31 | 67 | 87 | 91 => Some(d.Format),
        _ => None,
    }
}

#[cfg(all(test, feature = "diagnostics"))]
mod tests {
    use super::*;

    /// A half-red scene of one texel under an overlay of three: transparent,
    /// half-opaque green (premultiplied) and opaque blue.
    #[test]
    fn composes_a_scaled_premultiplied_plane_on_the_card() {
        let Some(host) = crate::host::host() else {
            eprintln!("skipping display-plane readback: no D3D12 device");
            return;
        };
        let format = DXGI_FORMAT_R8G8B8A8_UNORM;
        let mut composer = Composer::new(host, format).unwrap();
        let list = host.new_list().unwrap();
        let texture = |width: u64| -> ID3D12Resource {
            let mut resource = None;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: width,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            // SAFETY: plain object creation.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        None,
                        &mut resource,
                    )
                    .unwrap()
            };
            resource.unwrap()
        };
        let (scene, overlay, output) = (texture(1), texture(3), texture(3));
        let target = D3D12_RESOURCE_STATE_RENDER_TARGET;
        let read = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        let commands = &list.list;
        // SAFETY: an open list and live textures; the read-back buffer is
        // mapped after the list has run.
        unsafe {
            commands.ClearRenderTargetView(host.rtv(Some(&scene), None), &[0.5, 0.0, 0.0, 1.0], None);
            let overlay_view = host.rtv(Some(&overlay), None);
            let colors = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.5], [0.0, 0.0, 1.0, 1.0]];
            for (i, color) in colors.iter().enumerate() {
                let rect = windows::Win32::Foundation::RECT {
                    left: i as i32,
                    right: i as i32 + 1,
                    top: 0,
                    bottom: 1,
                };
                commands.ClearRenderTargetView(overlay_view, color, Some(&[rect]));
            }
            commands.ResourceBarrier(&[
                crate::host::transition(&scene, target, read),
                crate::host::transition(&overlay, target, read),
            ]);
            composer
                .record(host, 0, commands, [(&scene, format), (&overlay, format)], &output, (3, 1))
                .unwrap();
            commands.ResourceBarrier(&[crate::host::transition(&output, target, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            let (buffer, footprint) = crate::capture::read_back(&host.device, commands, &output).unwrap();
            list.close();
            host.execute(&[&list]);
            host.device.GetDeviceRemovedReason().unwrap();
            let mut mapped = std::ptr::null_mut();
            buffer.Map(0, None, Some(&mut mapped)).unwrap();
            let start = mapped.cast::<u8>().add(footprint.Offset as usize);
            let pixels: [u8; 12] = std::slice::from_raw_parts(start, 12).try_into().unwrap();
            buffer.Unmap(0, None);
            for (actual, expected) in pixels.iter().zip([128, 0, 0, 255, 64, 128, 0, 255, 0, 0, 255, 255]) {
                assert!(actual.abs_diff(expected) <= 1, "unexpected composite {pixels:?}");
            }
        }
    }
}

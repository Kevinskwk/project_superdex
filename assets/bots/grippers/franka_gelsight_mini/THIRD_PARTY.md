# Third-party asset provenance

The GelSight Mini meshes in `source/` are copied without geometric changes
from [MMintLab/hydroshear](https://github.com/MMintLab/hydroshear), commit
`f815b82fdf3451852acd918933020a82cede1f3b`, under the accompanying MIT license.

| File | SHA-256 |
| --- | --- |
| `panda_attach_gs_mini.obj` | `2dea2f4ba5c9830a71c71c0d852d48b622cdd6d03c06f679ef61738ec304fda6` |
| `panda_adapter_gsmini.obj` | `eeab1cff056784c459eda40816a632d5ecd0d7bbed3dce724f3e9d85f4b9a5f2` |
| `gsmini_shell_hollow_transformed.obj` | `22fc9cdab1bcc58d3ee4e9a980bd89803bcac914d6e0817ef567fd6394157e9e` |
| `gsmini_elastomer_transformed_both_sides.obj` | `6e1d91bc09bd7eb29aab9fc1e1472725fb2b9166fe8833c77562c5a78da04a27` |
| `gsmini_elastomer_transformed_enclosed.obj` | `20ce7f04752c0d6e92645b6d811cb619a3ba482bbc6cf82d72a3c6d13aba8541` |

`generated/gel_tet.mochi.json` is a tetrahedral derivative of the enclosed
elastomer mesh. `generated/housing_collision.mochi.json` combines the supplied
Panda attachment, finger adapter, and sensor shell in their shared transformed
sensor frame. The mirrored Franka mounting frames are derived from HydroShear's
`tacsl_franka_gelsight_mini.urdf` and adapted so both exposed gels face inward.

The Franka hand collision mesh is reused in place from
`assets/test/urdf/fr3v2_1_urdf`, which contains its own Apache-2.0 license and
provenance notice.

bl_info = {
    "name": "Absolute Scale",
    "author": "Martynas Žiemys",
    "version": (1, 0),
    "blender": (3, 6, 1),
    "location": "View3D search -> Absolute Scale, "\
                "View3D -> Mesh -> Transform -> Transform",
    "description": "Scales selected elements' bounds to specified dimensions",
    "warning": "",
    "doc_url": "",
    "category": "Object",
}

import bpy
import gpu
import bmesh
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from mathutils import Vector, Matrix
from bpy.props import FloatVectorProperty, BoolProperty

class MESH_OT_absolute_scale(bpy.types.Operator):
    """Scales selected elements' bounds to some dimensions"""
    bl_idname = "mesh.absolute_scale"
    bl_label = "Absolute Scale zzz"
    bl_options = {'REGISTER', 'UNDO'}
    
    a_scale: FloatVectorProperty(
        name="Absolute Scale",
        description="Scale in units",
        size = 3,
        subtype ='XYZ_LENGTH',
        default=Vector((1,1,1))
    ) 

    uniform: bpy.props.BoolProperty(
        name="Uniform",
        description="Scale all axes uniformly based on the reference axis",
        default=False
    )
    ref_axis: bpy.props.EnumProperty(
        name="Reference Axis",
        items=[
            ('X', 'X', ''),
            ('Y', 'Y', ''),
            ('Z', 'Z', ''),
        ],
        default='Y'
    )
    
    @classmethod
    def poll(cls, context):
        o = context.object
        if o is None:
            return False
        if context.mode != 'EDIT_MESH':
            return False
        return True
    
    def execute(self, context):
        bm = bmesh.from_edit_mesh(self.data)
        bm.verts.ensure_lookup_table()
        verts = [(self.M @ bm.verts[idx].co) @ self.O for idx in self.selected_verts]
        min_v = Vector(map(min, zip(*verts)))
        max_v = Vector(map(max, zip(*verts)))
        size = max_v - min_v
        target = Vector(self.a_scale)
        axis_index = {'X': 0, 'Y': 1, 'Z': 2}[self.ref_axis]
        ref_size = size[axis_index]
        ref_target = target[axis_index]
        ref_factor = ref_target / ref_size
        if self.uniform:
            scale = Vector((ref_factor, ref_factor, ref_factor))
        else:
            scale = Vector((
                target.x / size.x if size.x > 1e-6 else 1.0,
                target.y / size.y if size.y > 1e-6 else 1.0,
                target.z / size.z if size.z > 1e-6 else 1.0,
            ))
        S = Matrix.Diagonal((*scale, 1.0))
        T = (
            Matrix.Translation(self.pivot)
            @ self.O @ S @ self.O_inv
            @ Matrix.Translation(-self.pivot)
        )
        for v in bm.verts:
            if v.select:
                v_world = self.M @ v.co
                v_world = T @ v_world
                v.co = self.M_inv @ v_world
        bmesh.update_edit_mesh(self.data, loop_triangles=False)
        self.reset_to_normal(context)
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "uniform")
        if self.uniform:
            layout.prop(self, "ref_axis")
            layout.prop(
                self, "a_scale",
                 index={'X': 0, 'Y': 1, 'Z': 2}[self.ref_axis],
                 text = self.ref_axis + " scale")
        else:
            layout.prop(self, "a_scale")
    
    def modal(self, context, event):
        context.area.tag_redraw()
        if event.type in {'X', 'Y', 'Z'} and event.value == 'PRESS':
            self.active_axis = event.type
            self.ref_axis = self.active_axis
            self.u_in = self.axis_input[self.active_axis]
            self.numeric_active = bool(self.u_in)
            self.status(context)
            return {'RUNNING_MODAL'}
        
        if event.type in {'TAB','MIDDLEMOUSE' } and event.value == 'PRESS':
            order = ('X', 'Y', 'Z')
            i = order.index(self.active_axis)
            self.active_axis = order[(i + 1) % 3]
            self.ref_axis = self.active_axis
            self.u_in = self.axis_input[self.active_axis]
            self.numeric_active = bool(self.u_in)
            self.status(context)
            return {'RUNNING_MODAL'}
        
        if event.type == 'MOUSEMOVE' and not self.numeric_active:   
            self.mouse_2d = Vector((event.mouse_region_x, event.mouse_region_y))
            delta = (self.mouse_2d - self.pivot_2d).length
            factor = max(1e-6, 1.0 + (delta - self.saved_delta) * 0.007)
            if self.uniform:
                self.scale = [factor, factor, factor]
            else:
                self.scale[{'X': 0, 'Y': 1, 'Z': 2}[self.active_axis]] = factor
            self.apply_vertex_scale(context)
            self.saved_delta = delta
            return {'RUNNING_MODAL'}
        
        if event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'}:
            bpy.types.SpaceView3D.draw_handler_remove(
                self.overlay_handler, 'WINDOW'
                )
            self.reset_to_normal(context)
            self.execute(context)
            self.clear_status(context)
            context.workspace.status_text_set(None)
            return {'FINISHED'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.restore_initial_state(context)
            bpy.types.SpaceView3D.draw_handler_remove(
                self.overlay_handler, 'WINDOW'
                )
            self.reset_to_normal(context)
            self.clear_status(context)
            return {'CANCELLED'}
        
        if event.type == 'U' and event.value == 'PRESS':
            self.restore_initial_state(context)
            self.uniform = not self.uniform  
            self.apply_absolute_input(context)
            
            return {'RUNNING_MODAL'}
        elif event.value == 'PRESS':
            if event.type == 'BACK_SPACE':
                self.u_in = self.u_in[:-1]
                self.numeric_active = bool(self.u_in)
            elif event.unicode:
                self.u_in += event.unicode
                self.numeric_active = True
            self.axis_input[self.active_axis] = self.u_in
            self.apply_absolute_input(context)
            self.status(context)
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):  
        theme = context.preferences.themes[0].user_interface
        self.colors = (theme.axis_x, theme.axis_y, theme.axis_z)     
        self.was_normal = False
        self.obj = context.object
        self.data = self.obj.data
        bm = bmesh.from_edit_mesh(self.data)
        self.selected_verts = [v.index for v in bm.verts if v.select]
        self.initial_state = {i: bm.verts[i].co.copy() for i in self.selected_verts}
        if len(self.initial_state)<1:
            self.report({'WARNING'}, "Select some geometry first")
            return {'CANCELLED'}
        self.M = context.object.matrix_world
        self.M_inv = self.M.inverted()
        self.pivot = self.get_pivot(context, bm)
        self.pivot_2d = view3d_utils.location_3d_to_region_2d(
            context.region,
            context.region_data, 
            self.pivot)
        self.mouse_2d = Vector((event.mouse_region_x, event.mouse_region_y))
        self.saved_delta = (self.mouse_2d - self.pivot_2d).length 
        self.O = self.get_active_transform_orientation_matrix(context)     
        self.O_inv = self.O.inverted()   
        self.numeric_active = False
        self.u_in = ""
        self.shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self.scale = [1.0, 1.0, 1.0]
        self.axis_input = {'X': '', 'Y': '', 'Z': ''}
        self.active_axis = 'Y'
        self.uniform = False
        self.overlay_handler = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_bbox, 
            (context,), 
            'WINDOW', 
            'POST_PIXEL'
            )
        context.window_manager.modal_handler_add(self)   
        context.workspace.status_text_set_internal((
            "| X/Y/Z - axis | U - uniform | " 
            "TAB/MMB - cycle axis | LMB/Enter - confirm | "
            "RMB/Esc - cancel |"
             ))
        return {'RUNNING_MODAL'}
    
############################# Additional Functions ############################ 
        
    def restore_initial_state(self, context): 
        bm = bmesh.from_edit_mesh(self.data)
        for v in bm.verts:
            if v.index in self.initial_state:
                v.co = self.initial_state[v.index].copy()
        bmesh.update_edit_mesh(self.data, loop_triangles=False)
        
    def parse_unit_input(self, context, text):
        if not text:
            return None
        unit_settings = context.scene.unit_settings
        try:
            value = bpy.utils.units.to_value(
                unit_settings.system,
                'LENGTH',
                text,
                str_ref_unit=unit_settings.length_unit
                )
        except:
            value = None
        return value

    def get_oriented_bbox_size(self, context):
        bm = bmesh.from_edit_mesh(self.data)
        verts = [(self.M @ bm.verts[i].co) @ self.O for i in self.selected_verts]
        min_v = Vector(map(min, zip(*verts)))
        max_v = Vector(map(max, zip(*verts)))
        return max_v - min_v
    
    def apply_absolute_input(self, context):
        value = self.parse_unit_input(context, self.u_in)
        if value is None:
            return
        size = self.get_oriented_bbox_size(context)
        axis_index = {'X': 0, 'Y': 1, 'Z': 2}[self.active_axis]
        current = size[axis_index]
        factor = value / current
        if factor < 0.0001:
            return
        if self.uniform:
            self.scale = [factor, factor, factor]
        else:
            self.scale = [1.0, 1.0, 1.0]
            self.scale[axis_index] = factor
        self.apply_vertex_scale(context)

    def status(self, context):
        unit_settings = context.scene.unit_settings
        s = self.a_scale
        s_x = bpy.utils.units.to_string(unit_settings.system,'LENGTH', s.x)
        s_y = bpy.utils.units.to_string(unit_settings.system,'LENGTH', s.y)
        s_z = bpy.utils.units.to_string(unit_settings.system,'LENGTH', s.z)
        txt = self.u_in if self.u_in else "..."
        u = " ON" if self.uniform else " OFF"
        context.area.header_text_set(
            f"Uniform(U):{u}   "
            f"X:{s_x:<8} Y:{s_y:<8}  Z:{s_z:<8} | "
            f"{self.active_axis}: {txt}"
        )
        
    def clear_status(self, context):
        context.area.header_text_set(None)

    def get_active_transform_orientation_matrix(self, context):
        slot = context.scene.transform_orientation_slots[0]
        pivot = context.scene.tool_settings.transform_pivot_point
        type_name = slot.type  
        if slot.type == 'GLOBAL':
            return Matrix.Identity(4)
        elif slot.type == 'LOCAL':
            return self.M.to_3x3().to_4x4()
        elif slot.type == 'NORMAL':
            bm = bmesh.from_edit_mesh(self.data)
            if pivot == 'ACTIVE_ELEMENT':

                M_rot = self.M.to_quaternion().to_matrix().to_4x4()
                elem = bm.select_history.active
                if isinstance(elem, bmesh.types.BMVert):
                    z = elem.normal.normalized()
                    x = z.orthogonal()
                    y = z.cross(x).normalized()
                    x = y.cross(z).normalized()
                    return M_rot @ Matrix((x, y, z)).transposed().to_4x4()
                elif isinstance(elem, bmesh.types.BMEdge):
                    v1, v2 = elem.verts
                    y = (v2.co - v1.co).normalized()
                    z = (v1.normal + v2.normal).normalized()
                    z -= y * z.dot(y)
                    if z.length_squared < 1e-8:
                        z = x.orthogonal()
                    else:
                        z.normalize()
                    x = z.cross(y).normalized()
                    z = y.cross(x).normalized()
                    return M_rot @ Matrix((x, y, z)).transposed().to_4x4()
                elif isinstance(elem, bmesh.types.BMFace):
                    selected_verts = [v.select for v in bm.verts]
                    selected_edges = [e.select for e in bm.edges]
                    selected_faces = [f.select for f in bm.faces]
                    for f in bm.faces:
                        f.select_set(False)
                    elem.select_set(True)
                    bmesh.update_edit_mesh(self.data, loop_triangles=False, destructive=False)
                    bpy.ops.transform.create_orientation(
                        name="Normal",
                        use=True,
                        overwrite=True
                    )
                    for f, sel in zip(bm.faces, selected_faces):
                        f.select_set(sel)
                    for e, sel in zip(bm.edges, selected_edges):
                        e.select_set(sel)
                    for v, sel in zip(bm.verts, selected_verts):
                        v.select_set(sel)
                    bmesh.update_edit_mesh(self.data, loop_triangles=False, destructive=False)
                    self.was_normal=True
                    return slot.custom_orientation.matrix.to_4x4()
                if elem is None:
                    self.obj.update_from_editmode()
                    active_face = self.data.polygons.active
                    bm = bmesh.from_edit_mesh(self.data)
                    if active_face != -1:
                        elem = bm.faces[active_face]
                        if elem.select:
                            selected_verts = [v.select for v in bm.verts]
                            selected_edges = [e.select for e in bm.edges]
                            selected_faces = [f.select for f in bm.faces]
                            for f in bm.faces:
                                f.select_set(False)
                            elem.select_set(True)
                            bpy.ops.transform.create_orientation(
                                name="Normal",
                                use=True,
                                overwrite=True
                            )
                            bm = bmesh.from_edit_mesh(self.data)
                            for f, sel in zip(bm.faces, selected_faces):
                                f.select_set(sel)
                            for e, sel in zip(bm.edges, selected_edges):
                                e.select_set(sel)
                            for v, sel in zip(bm.verts, selected_verts):
                                v.select_set(sel)
                            bmesh.update_edit_mesh(self.data, loop_triangles=False, destructive=True)
                            self.was_normal=True
                            return slot.custom_orientation.matrix.to_4x4()
            bpy.ops.transform.create_orientation(use=True, name = "Normal")
            self.was_normal = True
            return slot.custom_orientation.matrix.to_4x4()
        elif slot.type == 'GIMBAL':
            # No idea what this is :D Maybe later
            return Matrix.Identity(4)
        elif slot.type == 'VIEW':
            return context.region_data.view_matrix.to_3x3().inverted().to_4x4()
        elif slot.type == 'CURSOR':
            return context.scene.cursor.matrix
        elif slot.type == 'PARENT':
            if self.obj.parent:
                return self.obj.parent.matrix_world
            else:
                return Matrix.Identity(4)
        elif slot.custom_orientation :
            return slot.custom_orientation.matrix.to_4x4()
        return Matrix.Identity(4)
    
    def get_pivot(self, context, bm):
        pivot = context.scene.tool_settings.transform_pivot_point
        if pivot == 'MEDIAN_POINT':
            cs = [self.M @ co for co in self.initial_state.values()]
            return sum(cs, Vector()) / len(cs)
        if pivot == 'CURSOR':
            return  context.scene.cursor.location.copy()
        if pivot == 'ACTIVE_ELEMENT':
            # Not exposed to API, some elaborate trickery is needed
            elem = bm.select_history.active
            if isinstance(elem, bmesh.types.BMVert):
                return self.M @ elem.co.copy()
            elif isinstance(elem, bmesh.types.BMEdge):
                return self.M @ ((elem.verts[0].co + elem.verts[1].co) * 0.5)
            elif isinstance(elem, bmesh.types.BMFace):
                return self.M @ (sum((v.co for v in elem.verts), Vector()) / len(elem.verts))
            # The elaborate trickery:
            if elem is None:
                self.obj.update_from_editmode()
                active_face = self.data.polygons.active
                bm = bmesh.from_edit_mesh(self.data)
                if active_face != -1:
                    bm.faces.ensure_lookup_table()
                    elem = bm.faces[active_face]
                    if elem.select:
                        return self.M @ elem.calc_center_median()       
        # Bounding box center in all other cases
        cs = [co for co in self.initial_state.values()]
        min_v = Vector((min(v.x for v in cs), min(v.y for v in cs), min(v.z for v in cs)))
        max_v = Vector((max(v.x for v in cs), max(v.y for v in cs), max(v.z for v in cs)))
        return self.M @ ((min_v + max_v) * 0.5)
    
    def reset_to_normal(self, context):
        if self.was_normal:
            bpy.ops.transform.delete_orientation()
            context.scene.transform_orientation_slots[0].type = 'NORMAL'
            self.was_normal = False

    def draw_bbox(self, context):
        bm = bmesh.from_edit_mesh(self.obj.data)
        sx, sy, sz = self.scale
        S = Matrix.Diagonal((sx, sy, sz, 1.0))
        O_scaled = self.O @ S
        verts = [(self.M @ bm.verts[idx].co) @ O_scaled for idx in self.selected_verts]
        min_v = Vector((min(v.x for v in verts),
                        min(v.y for v in verts),
                        min(v.z for v in verts)))
        max_v = Vector((max(v.x for v in verts),
                        max(v.y for v in verts),
                        max(v.z for v in verts)))
        corners = (
            Vector((min_v.x, min_v.y, min_v.z)),
            Vector((min_v.x, min_v.y, max_v.z)),
            Vector((min_v.x, max_v.y, min_v.z)),
            Vector((min_v.x, max_v.y, max_v.z)),
            Vector((max_v.x, min_v.y, min_v.z)),
            Vector((max_v.x, min_v.y, max_v.z)),
            Vector((max_v.x, max_v.y, min_v.z)),
            Vector((max_v.x, max_v.y, max_v.z)),
        )
        # messy bit | Would be nice to not do this while drawing... oh well...
        self.a_scale = Vector((max_v.x - min_v.x, max_v.y - min_v.y, max_v.z - min_v.z))
        region = context.region
        r_data = context.region_data
        corners_2d = []
        for c in corners:
            p = view3d_utils.location_3d_to_region_2d(region, r_data, O_scaled @ c)
            corners_2d.append(p)
        self.shader.bind()
        bbox_edges = (
            ((0, 4), (1, 5), (2, 6), (3, 7)),  # X
            ((1, 3), (2, 0), (5, 7), (6, 4)),  # Y
            ((0, 1), (7, 6), (3, 2), (4, 5)),  # Z
        )    
        for edges, color in zip(bbox_edges, self.colors):
            coords = []
            for a, b in edges:
                coords.extend((corners_2d[a], corners_2d[b]))
            batch = batch_for_shader(self.shader, 'LINES', {"pos": coords})
            self.shader.uniform_float("color", (*color, 1.0))
            batch.draw(self.shader)
        # Constraint line
        axis_index = {'X': 0, 'Y': 1, 'Z': 2}[self.active_axis]
        axis_dirs = (
            Vector((1, 0, 0)),
            Vector((0, 1, 0)),
            Vector((0, 0, 1)),
        )
        direction = self.O @ (axis_dirs[axis_index]).normalized() 
        p1 = self.pivot - direction 
        p2 = self.pivot + direction
        p1_2d = view3d_utils.location_3d_to_region_2d(region, r_data, p1)
        p2_2d = view3d_utils.location_3d_to_region_2d(region, r_data, p2)
        mid = (p1_2d + p2_2d) * 0.5
        dir_2d = (p2_2d - p1_2d).normalized()
        length = max(region.width, region.height)
        p_start = mid - dir_2d * length
        p_end   = mid + dir_2d * length
        thickness = 2
        dir_2d = (p_end - p_start).normalized()
        perp = Vector((-dir_2d.y, dir_2d.x)) * (thickness * 0.5)
        v1 = p_start + perp
        v2 = p_end   + perp
        v3 = p_end   - perp
        v4 = p_start - perp
        batch = batch_for_shader(
            self.shader,
            'TRIS',
            {"pos": (v1, v2, v3, v1, v3, v4)}
        )
        self.shader.bind()
        self.shader.uniform_float("color", (*self.colors[axis_index], 1.0))
        batch.draw(self.shader)
        # Line from pivot to mouse
        batch = batch_for_shader(
            self.shader,
            'LINES',
            {"pos": (self.pivot_2d, self.mouse_2d)}
        )
        self.shader.bind()
        self.shader.uniform_float("color", (0.7, 0.7, 0.7, 1.0))  
        batch.draw(self.shader)
        self.status(context)

    def apply_vertex_scale(self, context):
        bm = bmesh.from_edit_mesh(self.data)
        sx, sy, sz = self.scale
        S = Matrix.Diagonal((sx, sy, sz, 1.0))
        T_world = (
            Matrix.Translation(self.pivot)
            @ self.O @ S @ self.O_inv
            @ Matrix.Translation(-self.pivot)
        )
        for idx in self.selected_verts:
            v = bm.verts[idx]
            v_world = T_world @ self.M @ v.co
            v.co = self.M_inv @ v_world
        bmesh.update_edit_mesh(self.data, loop_triangles=False)
        self.scale = [1.0, 1.0, 1.0]
        
def absolute_scale_button(self, context):
    self.layout.operator(MESH_OT_absolute_scale.bl_idname, icon='PLUGIN')

def register():
    bpy.utils.register_class(MESH_OT_absolute_scale)
    bpy.types.VIEW3D_MT_transform.append(absolute_scale_button)

def unregister():
    bpy.types.VIEW3D_MT_transform.remove(absolute_scale_button)
    bpy.utils.unregister_class(MESH_OT_absolute_scale)

if __name__ == "__main__":
    register()
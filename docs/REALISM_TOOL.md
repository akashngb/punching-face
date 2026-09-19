# Realism tool

After loading a reconstructed head, choose **Realism → Enhance realism** in the
left panel. Processing stays in a browser worker on this computer. Adjust
**Strength**, switch between **Original** and **Enhanced**, or choose **Remove
realism pass** to recover the untouched texture. Rerunning starts from that
original; adjustments never accumulate on top of previous adjustments.

The pass reduces colour discontinuities left by photo cropping and texture
projection. It samples the texture onto the connected mesh, diffuses broad colour
along its edges, and bakes the resulting correction at the original resolution.
Fine texture is retained as a high-frequency residual. Separate UV islands cannot
contaminate each other's image filtering; connected geometry allows correction
across UV seams. Spatial proximity alone cannot connect unrelated surfaces.

Captured-head landmarks protect eyes, eyebrows, nose and mouth. The saved hair
roots and ear regions also protect those materials. Ordinary imported GLBs without
that reconstruction information receive surface colour balancing without the same
semantic feature protection. Overlapping/repeating UV layouts are unsupported.

This is an appearance estimate, not a generative reconstruction or a recovery of
missing anatomy. It does not move vertices, change topology, recover pores that
were never captured, fix an incorrect silhouette, or repair missing ears/hair.
Start with moderate strength and inspect the profile as well as the front.

**Export GLB** bakes whichever comparison view is currently selected. **Save
editable session** retains the current view, original texture, strength, and
repair metadata; reopening the session recreates the reversible pass. The saved
reconstruction on the server is never overwritten by the preview. Geometry,
physics bindings, expressions and accessories remain available.

Implementation: `src/realism-surface.js` (pure processing),
`src/realism-worker.js` (background execution), and `src/realism-controls.js`
(preview/lifecycle). Run `node --test tests/realism.test.mjs` for seam continuity,
disconnected surface isolation, protected regions, alpha, UV orientation, and
invalid-input regression checks.

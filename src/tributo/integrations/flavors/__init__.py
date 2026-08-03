"""Built-in bundle model flavors (loader implementations).

Flavors implement the ``BundleModelFlavor`` protocol from
``tributo.exporting.runtime``: each one knows how to load a
``ResolvedArtifact`` into an in-memory ``BundleModel``.  The registry
is populated from the ``tributo.model_flavors`` entry points; the
serveable matrix lives in ``exporting.runtime.SERVEABLE_FLAVOR_MATRIX``.
"""

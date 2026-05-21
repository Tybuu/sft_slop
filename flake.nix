{
  description = "CSE481L Flake";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  # nixConfig = {
  #   extra-substituters = [
  #     "https://cuda-maintainers.cachix.org"
  #   ];
  #   extra-trusted-public-keys = [
  #     "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
  #   ];
  # };

  outputs = {
    self,
    nixpkgs,
    ...
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };
  in {
    devShells.x86_64-linux.default = pkgs.mkShell {
      packages = [
        (pkgs.python3)
        pkgs.python3Packages.pip
        pkgs.cudaPackages.cudatoolkit
      ];

      shellHook = ''
        export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
        export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.libglvnd
          pkgs.glib
          pkgs.zlib
        ]}:$LD_LIBRARY_PATH"
        if [ ! -d .venv ]; then
          python -m venv .venv
        fi
        source .venv/bin/activate
      '';
    };
  };
}

# shell.nix in your MLIiP_2026 project root
{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
	buildInputs = [
		pkgs.python313
		pkgs.uv
		pkgs.cudaPackages.cudatoolkit
		pkgs.btop
	];

	shellHook = ''
		export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.cudaPackages.cudatoolkit}/lib:$LD_LIBRARY_PATH
		export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
	'';
}

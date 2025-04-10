using LinearAlgebra

function load_distance_matrix(filepath)
    lines = readlines(filepath)
    species = split(lines[1])[2:end]  # skip the initial number (119)
    n = length(species)
    mat = zeros(n, n)
    
    for (i, line) in enumerate(lines[2:end])
        parts = split(line)
        values = parse.(Float64, parts[2:end])  # skip species name
        mat[i, :] = values
    end
    return species, mat
end

species, distmat = load_distance_matrix("data/phylo_trees/allspeciesList_distmat.txt")
# the matrix is symmetric 
issymmetric(distmat)

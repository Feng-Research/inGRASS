cd("../data/")
ar_org, W, A = mtx2arWG("G2_circuit.mtx")

ext, W_add = Rext("extG2.mtx")

Random.seed!(1234); randstring()
W_add = ones(Int, length(W_add)) .* rand(1:2, length(W_add))

arP, WP, AP = mtx2arW("G2_SP.mtx")
cd("../src/")

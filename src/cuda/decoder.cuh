
#include <cstdint>

template <typename OP>
__global__ void FirstEdge(
    size_t* d_indptr,
    uint8_t* d_indices,
    uint32_t v,
    OP op
) {
    //*****************************************************************//
    // https://github.com/jshun/ligra/blob/master/ligra/byteRLE.h#L64-L83
    auto offset = d_indptr[v];
            
    auto const fb = d_indices[offset];
    offset++;

    auto edge_read = fb & 0x3f;
    if (fb & 0x80) {
        int shift_amount = 6;
        while (true) {
            auto b = d_indices[offset];
            edge_read |= (b & 0x7f) << shift_amount;
            offset++;
            if (b & 0x80) {
                shift_amount += 7;
            } else {
                break;
            }
        }
    }

    uint32_t first_edge = (fb & 0x40) ? v - edge_read : v + edge_read;
    //*****************************************************************//
    
    op(first_edge);

    NextEdges<<<1, 64>>>(d_indptr, d_indices, v, offset, first_edge, op);
}

template <typename OP>
__global__ void NextEdges(
    size_t* d_indptr,
    uint8_t* d_indices,
    uint32_t v,
    size_t start,
    uint32_t prev_edge,
    OP op
) {
    auto offset = start;

    auto const end = d_indptr[v];

    while (offset < end) {
        auto header = d_indices[offset];
        offset++;
	auto num_bytes = 1 + (header & 0x3);
        auto run_length = 1 + (header >> 2);
        



template <typename OP>
__global__ void adj_map_kernel(
    size_t* d_indptr,
    uint8_t* d_indices,
    uint v,
    OP op
) {
    // designate this group
    auto tb = cg::this_thread_block();

    __shared__ uint prev_edge;

    __shared__ size_t start; 
    start = d_indptr[v];
    __shared__ size_t end;
   end = d_indptr[v + 1];

    tb.sync();

    if ((end - start) > 0) { 

        // choose a leader to compute the first edge
        if (tb.thread_rank() == 0) {

        }

        tb.sync();
		    
	__shared__ uint num_bytes;
        __shared__ uint run_length;
	__shared__ uint smem_buffer[64];

	while (start < end) {
	    // parse header
            if (tb.thread_rank() == 0) {
                uchar header = d_indices[start];
	        start++;
	        num_bytes = 1 + (header & 0x3);
                run_length = 1 + (header >> 2);
            }
            tb.sync();

            // compute diffs
            uint diff = 0; 
            for (int i = 0; i < num_bytes; i++) {
                diff = diff << 8;
                diff += (uint) d_indices[start + (threadIdx.x * num_bytes) + i];
            }
            smem_buffer[threadIdx.x] = prev_edge + diff;
            tb.sync();

            auto tile = cg::tiled_partition<8>(tb);

            uint u = cg::inclusive_scan(tile, smem_buffer);

            op(u);

            if (tb.thread_rank() == (run_length - 1)) {
                start += (num_bytes * run_length);
                prev_edge = smem_buffer[threadIdx.x];
            }
            tb.sync();
	}
    }
}


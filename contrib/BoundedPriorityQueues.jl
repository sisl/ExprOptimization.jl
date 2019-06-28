# *****************************************************************************
# Written by Ritchie Lee, ritchie.lee@sv.cmu.edu
# *****************************************************************************
# Copyright ã 2015, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration. All
# rights reserved.  The Reinforcement Learning Encounter Simulator (RLES)
# platform is licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You
# may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable
# law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
# _____________________________________________________________________________
# Reinforcement Learning Encounter Simulator (RLES) includes the following
# third party software. The SISLES.jl package is licensed under the MIT Expat
# License: Copyright (c) 2014: Youngjun Kim.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
# "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# *****************************************************************************

module BoundedPriorityQueues

export BoundedPriorityQueue, enqueue! 

using DataStructures

mutable struct BoundedPriorityQueue{K,V}
    pq::PriorityQueue{K,V}
    N::Int64

    function BoundedPriorityQueue{K,V}(N::Int64, o::Base.Order.Ordering=Base.Order.Forward) where {K,V}
        #higher is kept
        new{K,V}(PriorityQueue{K,V}(o), N)
    end
end

function DataStructures.enqueue!(q::BoundedPriorityQueue{K,V}, k::K, v::V;
    make_copy::Bool=false) where {K,V}
    haskey(q.pq, k) && return -1 #keys must be unique, return -1 if collision
    if make_copy
        k = deepcopy(k)
    end
    n = length(q.pq)
    enqueue!(q.pq, k, v)
    n_add = n - length(q.pq) #number of items added
    while length(q.pq) > q.N
        dequeue!(q.pq)
    end
    n_add
end

Base.length(q::BoundedPriorityQueue) = length(q.pq)
Base.isempty(q::BoundedPriorityQueue) = isempty(q.pq) 
Base.haskey(q::BoundedPriorityQueue) = haskey(q.pq)
Base.keys(q::BoundedPriorityQueue) = keys(q.pq)
Base.values(q::BoundedPriorityQueue) = values(q.pq)
function Base.empty!(q::BoundedPriorityQueue)
    while !isempty(q.pq)
        dequeue!(q.pq)
    end
end

"""
Ordered iterator
"""
function Base.iterate(q::BoundedPriorityQueue) 
    pairs = collect(q.pq)
    sort!(pairs, by=x->x[2], rev=(q.pq.o==Base.Order.ForwardOrdering()))
    result = iterate(pairs)
    if result == nothing
        return nothing
    else
        item, state = result
        return item, (pairs=pairs, state=state)
    end
end
function Base.iterate(q::BoundedPriorityQueue, x) 
    result = iterate(x.pairs, x.state) 
    if result == nothing
        return nothing
    else
        item, state = result
        return item, (pairs=x.pairs, state=state)
    end
end

end #module

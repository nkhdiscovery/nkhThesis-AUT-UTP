/*
Copyright (C) 2015 Yasutomo Kawanishi
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#include "segment-graph.h"
#include <iostream>
#include <opencv2/core.hpp>

bool operator<(const edge &a, const edge &b) {
  return a.w < b.w;
}


float nkhThresh(universe *u, int a, float c)
{

    if(u->size(a)==2)
    {
        std::cout<< a << std::endl;
    }
    return THRESHOLD(u->size(a), c);
}

/*
 * Segment a graph
 *
 * Returns a disjoint-set forest representing the segmentation.
 *
 * num_vertices: number of vertices in graph.
 * num_edges: number of edges in graph
 * edges: array of edges.
 * c: constant for treshold function.
 */
universe *segment_graph(int w, int h, int num_edges, edge *edges,
            float c) {
  // sort edges by weight
  std::sort(edges, edges + num_edges);
  int num_vertices = w * h;
  // make a disjoint-set forest
  universe *u = new universe(w, h);

  // init thresholds
  float *threshold = new float[num_vertices];
  for (int i = 0; i < num_vertices; i++)
    threshold[i] = THRESHOLD(1,c);

  // for each edge, in non-decreasing weight order...
  for (int i = 0; i < num_edges; i++) {
    edge *pedge = &edges[i];
    
    // components conected by this edge
    int a = u->find(pedge->a);
    int b = u->find(pedge->b);
    if (a != b) {
      if ((pedge->w <= threshold[a]) &&
	  (pedge->w <= threshold[b])) {
	u->join(a, b);
	a = u->find(a);
    threshold[a] = pedge->w + nkhThresh(u, a, c);
//    std::cout << threshold[a] << std::endl;
   // std::cout << pedge->w << ", " << THRESHOLD(u->size(a), c) << std::endl;
      }
    }
  }

  // free up
  delete threshold;
  return u;
}

